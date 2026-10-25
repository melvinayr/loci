import Foundation
import MachO
import Darwin.malloc
import os

private let log = Logger(subsystem: "dev.loci", category: "residency")

public enum ModelRole: String, Sendable, Hashable, CaseIterable {
    case embedder, reranker, generator
}

/// Which models may stay loaded together.
public enum ResidencyPolicy: Sendable, Hashable {
    /// One model resident at a time; loading another unloads the previous one.
    case lowMemory
    /// Embedder and reranker stay resident; the generator is loaded for one answer and released.
    case balanced
    /// Everything stays resident.
    case keepResident
}

/// How much memory the process may still allocate.
public protocol MemoryProbe: Sendable {
    func availableBytes() -> UInt64
}

/// Real memory: `os_proc_available_memory()` on iOS (the budget before the system terminates the
/// app); physical RAM minus the app's current footprint on macOS.
public struct SystemMemoryProbe: MemoryProbe {
    public init() {}

    public func availableBytes() -> UInt64 {
        #if os(iOS)
        return UInt64(os_proc_available_memory())
        #else
        let physical = ProcessInfo.processInfo.physicalMemory
        let used = Self.currentFootprint()
        return physical > used ? physical - used : 0
        #endif
    }

    static func currentFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }
}

/// The app's own memory use, as reported by the kernel.
public enum ProcessMemory {
    /// Bytes the process is currently charged for (`phys_footprint`).
    public static var footprintBytes: UInt64 { SystemMemoryProbe.currentFootprint() }

    /// The highest footprint the process has reached so far, which can predate the current operation.
    public static var peakFootprintBytes: UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? max(UInt64(info.ledger_phys_footprint_peak), info.phys_footprint) : 0
    }
}

/// Loads models on demand, keeps them according to a `ResidencyPolicy`, refuses loads that would not
/// fit, and unloads idle models under memory pressure.
///
/// Callers borrow a model with `use(_:_:)`; a model is never unloaded while borrowed. Do not nest
/// `use` calls for different roles: under `.lowMemory` the inner call waits for the outer one to finish.
public actor ModelResidencyManager {
    private struct Registration {
        var estimatedBytes: @Sendable () async throws -> UInt64
        var load: @Sendable () async throws -> any Sendable & AnyObject
    }

    private struct Resident {
        var model: any Sendable & AnyObject
        var bytes: UInt64
        var lastUsed: UInt64
    }

    public let policy: ResidencyPolicy
    private let memory: any MemoryProbe
    private var registrations: [ModelRole: Registration] = [:]
    private var resident: [ModelRole: Resident] = [:]
    private var borrowers: [ModelRole: Int] = [:]
    private var unloadWhenIdle: Set<ModelRole> = []
    private var clock: UInt64 = 0

    private var loadLocked = false
    private var loadWaiters: [CheckedContinuation<Void, Never>] = []
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []
    // nonisolated(unsafe) is sound here: the source is assigned once, in `init` before the actor can be
    // shared, and afterwards only cancelled from `deinit`; `DispatchSource.cancel()` is thread-safe.
    private nonisolated(unsafe) var pressureSource: (any DispatchSourceMemoryPressure)?

    /// - Parameter observesMemoryPressure: unload idle models when the system reports memory pressure.
    public init(policy: ResidencyPolicy, memory: any MemoryProbe = SystemMemoryProbe(), observesMemoryPressure: Bool = true) {
        self.policy = policy
        self.memory = memory
        if observesMemoryPressure { startObservingMemoryPressure() }
    }

    deinit {
        pressureSource?.cancel()
    }

    /// Declares how to size and load the model for `role`, replacing any earlier registration
    /// (and unloading the old model).
    public func register<Model: Sendable & AnyObject>(
        _ role: ModelRole,
        estimatedBytes: @escaping @Sendable () async throws -> UInt64,
        load: @escaping @Sendable () async throws -> Model
    ) {
        unload(role)
        registrations[role] = Registration(estimatedBytes: estimatedBytes, load: { try await load() })
    }

    public func isRegistered(_ role: ModelRole) -> Bool { registrations[role] != nil }

    /// Roles whose models are currently loaded.
    public var residentRoles: Set<ModelRole> { Set(resident.keys) }

    /// Sum of the load-time estimates of loaded models.
    public var residentBytes: UInt64 { resident.values.reduce(0) { $0 + $1.bytes } }

    /// Runs `body` with the model for `role`, loading it first if needed.
    public func use<Model: Sendable & AnyObject, Result: Sendable>(
        _ role: ModelRole, as type: Model.Type = Model.self,
        _ body: @Sendable (Model) async throws -> Result
    ) async throws -> Result {
        guard registrations[role] != nil else {
            throw LlamaError.backend(operation: "no \(role.rawValue) model is configured", code: -1)
        }
        if policy == .lowMemory { await waitUntilOthersAreIdle(than: role) }
        let loaded = try await ensureLoaded(role)
        guard let model = loaded as? Model else {
            throw LlamaError.backend(operation: "\(role.rawValue) model has an unexpected type", code: -1)
        }

        borrowers[role, default: 0] += 1
        defer { release(role) }
        return try await body(model)
    }

    /// Unloads `role` now, or as soon as it is no longer borrowed.
    public func unload(_ role: ModelRole) {
        if (borrowers[role] ?? 0) > 0 {
            unloadWhenIdle.insert(role)
        } else {
            resident[role] = nil
        }
    }

    public func unloadAll() {
        for role in ModelRole.allCases { unload(role) }
    }

    /// Unloads every model that is not borrowed; borrowed ones go as soon as they are released.
    public func unloadIdle() {
        for role in resident.keys {
            unload(role)
        }
    }

    /// The response to a system memory warning.
    func handleMemoryPressure() {
        log.notice("memory pressure: unloading \(self.resident.count) models")
        unloadIdle()
    }

    // MARK: Loading

    private func ensureLoaded(_ role: ModelRole) async throws -> any Sendable & AnyObject {
        if var existing = resident[role] {
            clock += 1
            existing.lastUsed = clock
            resident[role] = existing
            return existing.model
        }

        await lockLoading()
        defer { unlockLoading() }
        if let existing = resident[role] { return existing.model }  // loaded while we waited
        guard let registration = registrations[role] else {
            throw LlamaError.backend(operation: "no \(role.rawValue) model is configured", code: -1)
        }

        let needed = try await registration.estimatedBytes()
        try await makeRoom(for: role, bytes: needed)
        let model = try await registration.load()
        clock += 1
        resident[role] = Resident(model: model, bytes: needed, lastUsed: clock)
        log.info("loaded \(role.rawValue, privacy: .public), estimated \(needed) bytes")
        return model
    }

    /// Applies the policy, then evicts idle models (least recently used first) until the load fits.
    private func makeRoom(for role: ModelRole, bytes needed: UInt64) async throws {
        if policy == .lowMemory {
            for other in resident.keys where other != role { resident[other] = nil }
        }

        var available = memory.availableBytes()
        if needed > available {
            // Evicted models free their weights and buffers, but the allocator keeps freed pages until asked; on iOS those
            // pages still count against the app's budget, so hand them back before judging whether the load fits.
            releaseFreedMemory()
            available = memory.availableBytes()
        }
        while needed > available {
            // Keep only the key: holding the entry would keep the model alive past its eviction.
            let victim = resident
                .filter { $0.key != role && (borrowers[$0.key] ?? 0) == 0 }
                .min { $0.value.lastUsed < $1.value.lastUsed }?.key
            guard let victim else {
                throw LlamaError.insufficientMemory(required: needed, available: available)
            }
            log.notice("evicting \(victim.rawValue, privacy: .public) to make room for \(role.rawValue, privacy: .public)")
            resident[victim] = nil
            releaseFreedMemory()
            available = memory.availableBytes()
        }
    }

    private func releaseFreedMemory() {
        let before = memory.availableBytes()
        _ = malloc_zone_pressure_relief(nil, 0)
        log.info("returned freed memory to the system: available \(before) -> \(self.memory.availableBytes()) bytes")
    }

    // MARK: Borrowing

    private func release(_ role: ModelRole) {
        borrowers[role, default: 1] -= 1
        guard (borrowers[role] ?? 0) <= 0 else { return }
        borrowers[role] = nil
        if unloadWhenIdle.remove(role) != nil || (policy == .balanced && role == .generator) {
            resident[role] = nil
        }
        let waiting = idleWaiters
        idleWaiters = []
        for waiter in waiting { waiter.resume() }
    }

    /// Under `.lowMemory`, waits until no other role is borrowed, so two models never overlap.
    private func waitUntilOthersAreIdle(than role: ModelRole) async {
        while borrowers.contains(where: { $0.key != role && $0.value > 0 }) {
            await withCheckedContinuation { idleWaiters.append($0) }
        }
    }

    private func lockLoading() async {
        if !loadLocked {
            loadLocked = true
            return
        }
        await withCheckedContinuation { loadWaiters.append($0) }
    }

    private func unlockLoading() {
        if loadWaiters.isEmpty {
            loadLocked = false
        } else {
            loadWaiters.removeFirst().resume()  // the lock passes straight to the next waiter
        }
    }

    // MARK: Memory pressure

    private nonisolated func startObservingMemoryPressure() {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .global(qos: .utility))
        source.setEventHandler { [weak self] in
            guard let self else { return }
            Task { await self.handleMemoryPressure() }
        }
        source.resume()
        pressureSource = source
    }
}
