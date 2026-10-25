import Foundation
import os
import Testing
@testable import LociLlama

/// A model whose lifetime is visible: it adds its size to a ledger while alive.
final class FakeModel: Sendable {
    let name: String
    private let ledger: MemoryLedger
    private let bytes: UInt64

    init(name: String, bytes: UInt64, ledger: MemoryLedger) {
        self.name = name
        self.bytes = bytes
        self.ledger = ledger
        ledger.allocate(bytes)
    }

    deinit { ledger.free(bytes) }
}

/// Memory that shrinks as fake models load and recovers when they are freed, like the real thing.
final class MemoryLedger: MemoryProbe {
    private let state: OSAllocatedUnfairLock<(capacity: UInt64, used: UInt64)>

    init(capacity: UInt64) { state = OSAllocatedUnfairLock(initialState: (capacity, 0)) }

    func availableBytes() -> UInt64 { state.withLock { $0.capacity > $0.used ? $0.capacity - $0.used : 0 } }
    func allocate(_ bytes: UInt64) { state.withLock { $0.used += bytes } }
    func free(_ bytes: UInt64) { state.withLock { $0.used -= bytes } }
    var used: UInt64 { state.withLock { $0.used } }
    func setCapacity(_ value: UInt64) { state.withLock { $0.capacity = value } }
}

/// Counts loads per role.
final class LoadLog: Sendable {
    private let events = OSAllocatedUnfairLock(initialState: [String]())
    func record(_ role: ModelRole) { events.withLock { $0.append(role.rawValue) } }
    var all: [String] { events.withLock { $0 } }
    func count(_ role: ModelRole) -> Int { all.filter { $0 == role.rawValue }.count }
}

private func makeManager(
    _ policy: ResidencyPolicy, capacity: UInt64 = 10_000, sizes: [ModelRole: UInt64] = [:],
    roles: [ModelRole] = ModelRole.allCases
) async -> (ModelResidencyManager, MemoryLedger, LoadLog) {
    let ledger = MemoryLedger(capacity: capacity)
    let log = LoadLog()
    let manager = ModelResidencyManager(policy: policy, memory: ledger, observesMemoryPressure: false)
    for role in roles {
        let bytes = sizes[role] ?? 100
        await manager.register(role, estimatedBytes: { bytes }, load: {
            log.record(role)
            return FakeModel(name: role.rawValue, bytes: bytes, ledger: ledger)
        })
    }
    return (manager, ledger, log)
}

private func touch(_ manager: ModelResidencyManager, _ role: ModelRole) async throws -> String {
    try await manager.use(role, as: FakeModel.self) { $0.name }
}

@Suite("Model residency: loading")
struct ResidencyLoadingTests {
    @Test func modelsLoadLazilyAndAreReused() async throws {
        let (manager, _, log) = await makeManager(.keepResident)
        #expect(await manager.residentRoles.isEmpty)
        #expect(try await touch(manager, .embedder) == "embedder")
        _ = try await touch(manager, .embedder)
        _ = try await touch(manager, .embedder)
        #expect(log.count(.embedder) == 1)
        #expect(await manager.residentRoles == [.embedder])
        #expect(await manager.residentBytes == 100)
    }

    @Test func concurrentFirstUsesLoadOnce() async throws {
        let (manager, _, log) = await makeManager(.keepResident)
        try await withThrowingTaskGroup(of: String.self) { group in
            for _ in 0..<12 { group.addTask { try await touch(manager, .reranker) } }
            for try await _ in group {}
        }
        #expect(log.count(.reranker) == 1)
    }

    @Test func aFailedLoadLeavesNothingResidentAndDoesNotWedgeLaterLoads() async throws {
        struct Boom: Error {}
        let manager = ModelResidencyManager(policy: .keepResident, memory: MemoryLedger(capacity: 1_000), observesMemoryPressure: false)
        let attempts = OSAllocatedUnfairLock(initialState: 0)
        await manager.register(.embedder, estimatedBytes: { 10 }, load: {
            if attempts.withLock({ $0 += 1; return $0 }) == 1 { throw Boom() }
            return FakeModel(name: "ok", bytes: 0, ledger: MemoryLedger(capacity: 0))
        })
        await #expect(throws: Boom.self) { try await touch(manager, .embedder) }
        #expect(await manager.residentRoles.isEmpty)
        #expect(try await touch(manager, .embedder) == "ok")
    }

    @Test func unregisteredRolesAndWrongTypesAreErrorsNotCrashes() async throws {
        let (manager, _, _) = await makeManager(.keepResident, roles: [.embedder])
        await #expect(throws: LlamaError.self) { try await touch(manager, .generator) }
        await #expect(throws: LlamaError.self) {
            try await manager.use(.embedder, as: Embedder.self) { _ in 0 }
        }
        let r1 = await manager.isRegistered(.embedder)
        #expect(r1)
        let r2 = await manager.isRegistered(.generator)
        #expect(!r2)
    }

    @Test func registeringAgainReplacesAndUnloadsTheOldModel() async throws {
        let (manager, ledger, _) = await makeManager(.keepResident, roles: [.embedder])
        _ = try await touch(manager, .embedder)
        #expect(ledger.used == 100)
        await manager.register(.embedder, estimatedBytes: { 50 }, load: { FakeModel(name: "new", bytes: 50, ledger: ledger) })
        #expect(ledger.used == 0)
        let r3 = await manager.residentRoles.isEmpty
        #expect(r3)
        #expect(try await touch(manager, .embedder) == "new")
    }
}

@Suite("Model residency: policies")
struct ResidencyPolicyTests {
    @Test func lowMemoryKeepsOneModelAtATime() async throws {
        let (manager, ledger, log) = await makeManager(.lowMemory)
        _ = try await touch(manager, .embedder)
        #expect(await manager.residentRoles == [.embedder])
        _ = try await touch(manager, .reranker)
        #expect(await manager.residentRoles == [.reranker], "the embedder was unloaded to make room")
        #expect(ledger.used == 100, "its memory was actually freed")
        _ = try await touch(manager, .generator)
        #expect(await manager.residentRoles == [.generator])
        _ = try await touch(manager, .reranker)
        #expect(log.all == ["embedder", "reranker", "generator", "reranker"], "the ask flow reloads the reranker for verification")
    }

    @Test func lowMemoryKeepsTheLastModelForRepeatedUse() async throws {
        let (manager, _, log) = await makeManager(.lowMemory)
        for _ in 0..<5 { _ = try await touch(manager, .embedder) }  // e.g. embedding batches during ingest
        #expect(log.count(.embedder) == 1)
    }

    @Test func balancedKeepsEmbedderAndRerankerAndReleasesTheGeneratorAfterUse() async throws {
        let (manager, ledger, log) = await makeManager(.balanced)
        _ = try await touch(manager, .embedder)
        _ = try await touch(manager, .reranker)
        #expect(await manager.residentRoles == [.embedder, .reranker])
        _ = try await touch(manager, .generator)
        #expect(await manager.residentRoles == [.embedder, .reranker], "the generator is on demand")
        #expect(ledger.used == 200)
        _ = try await touch(manager, .generator)
        #expect(log.count(.generator) == 2)
        #expect(log.count(.embedder) == 1 && log.count(.reranker) == 1)
    }

    @Test func keepResidentKeepsEverything() async throws {
        let (manager, _, log) = await makeManager(.keepResident)
        for role in ModelRole.allCases { _ = try await touch(manager, role) }
        for role in ModelRole.allCases { _ = try await touch(manager, role) }
        #expect(await manager.residentRoles == Set(ModelRole.allCases))
        #expect(log.all.count == 3)
    }

    @Test func lowMemoryNeverUnloadsAModelThatIsBorrowed() async throws {
        let (manager, ledger, log) = await makeManager(.lowMemory)
        let (gate, open) = AsyncStream<Void>.makeStream()
        let holding = OSAllocatedUnfairLock(initialState: false)

        let first = Task {
            try await manager.use(.embedder, as: FakeModel.self) { model in
                holding.withLock { $0 = true }
                for await _ in gate { break }
                return model.name
            }
        }
        while !holding.withLock({ $0 }) { await Task.yield() }

        let second = Task { try await touch(manager, .reranker) }
        try await Task.sleep(for: .milliseconds(150))
        #expect(log.count(.reranker) == 0, "the reranker must wait while the embedder is in use")
        let r4 = await manager.residentRoles
        #expect(r4 == [.embedder])
        #expect(ledger.used == 100)

        open.yield()
        #expect(try await first.value == "embedder")
        #expect(try await second.value == "reranker")
        #expect(await manager.residentRoles == [.reranker])
    }
}

@Suite("Model residency: memory")
struct ResidencyMemoryTests {
    @Test func aLoadThatCannotFitIsRefusedBeforeLoading() async throws {
        let (manager, _, log) = await makeManager(.keepResident, capacity: 100, sizes: [.generator: 150])
        do {
            _ = try await touch(manager, .generator)
            Issue.record("expected insufficientMemory")
        } catch LlamaError.insufficientMemory(let required, let available) {
            #expect(required == 150 && available == 100)
        }
        #expect(log.count(.generator) == 0, "nothing was loaded")
        #expect(await manager.residentRoles.isEmpty)
    }

    @Test func afterAFailureTheLoadSucceedsOnceMemoryIsAvailable() async throws {
        let (manager, ledger, _) = await makeManager(.keepResident, capacity: 100, sizes: [.generator: 150])
        await #expect(throws: LlamaError.self) { try await touch(manager, .generator) }
        ledger.setCapacity(1_000)
        #expect(try await touch(manager, .generator) == "generator")
    }

    @Test func idleModelsAreEvictedLeastRecentlyUsedFirstToMakeRoom() async throws {
        let (manager, ledger, log) = await makeManager(.keepResident, capacity: 250)
        _ = try await touch(manager, .embedder)
        _ = try await touch(manager, .reranker)
        _ = try await touch(manager, .embedder)  // the reranker is now the least recently used
        #expect(ledger.availableBytes() == 50)

        _ = try await touch(manager, .generator)  // needs 100, only 50 free
        #expect(await manager.residentRoles == [.embedder, .generator], "the LRU model (reranker) was evicted")
        #expect(log.count(.embedder) == 1)
    }

    @Test func aBorrowedModelIsNotEvictedAndTheLoadFailsInstead() async throws {
        let (manager, _, _) = await makeManager(.keepResident, capacity: 150)
        let (gate, open) = AsyncStream<Void>.makeStream()
        let holding = OSAllocatedUnfairLock(initialState: false)
        let borrower = Task {
            try await manager.use(.embedder, as: FakeModel.self) { _ in
                holding.withLock { $0 = true }
                for await _ in gate { break }
            }
        }
        while !holding.withLock({ $0 }) { await Task.yield() }

        await #expect(throws: LlamaError.self) { try await touch(manager, .reranker) }
        #expect(await manager.residentRoles == [.embedder], "the borrowed model survived")
        open.yield()
        try await borrower.value
    }

    @Test func unloadingABorrowedModelWaitsUntilItIsReleased() async throws {
        let (manager, ledger, _) = await makeManager(.keepResident)
        let (gate, open) = AsyncStream<Void>.makeStream()
        let holding = OSAllocatedUnfairLock(initialState: false)
        let borrower = Task {
            try await manager.use(.embedder, as: FakeModel.self) { _ in
                holding.withLock { $0 = true }
                for await _ in gate { break }
            }
        }
        while !holding.withLock({ $0 }) { await Task.yield() }

        await manager.unload(.embedder)
        let r5 = await manager.residentRoles
        #expect(r5 == [.embedder])
        #expect(ledger.used == 100, "still borrowed")
        open.yield()
        try await borrower.value
        let r6 = await manager.residentRoles.isEmpty
        #expect(r6)
        #expect(ledger.used == 0)
    }

    @Test func memoryPressureUnloadsIdleModelsAndTheyReloadOnDemand() async throws {
        let (manager, ledger, log) = await makeManager(.keepResident)
        _ = try await touch(manager, .embedder)
        _ = try await touch(manager, .reranker)
        await manager.handleMemoryPressure()
        let r7 = await manager.residentRoles.isEmpty
        #expect(r7)
        #expect(ledger.used == 0)
        _ = try await touch(manager, .embedder)
        #expect(log.count(.embedder) == 2)
    }

    @Test func memoryPressureDuringUseUnloadsTheModelAsSoonAsItIsReleased() async throws {
        let (manager, ledger, _) = await makeManager(.keepResident)
        try await manager.use(.embedder, as: FakeModel.self) { _ in
            await manager.handleMemoryPressure()
            #expect(await manager.residentRoles == [.embedder], "in use, so not yet unloaded")
        }
        let r8 = await manager.residentRoles.isEmpty
        #expect(r8)
        #expect(ledger.used == 0)
    }

    @Test func theSystemMemorySourceCanBeObservedAndTornDownSafely() async throws {
        for _ in 0..<3 {
            let manager = ModelResidencyManager(policy: .balanced)  // observes real memory pressure
            await manager.register(.embedder, estimatedBytes: { 0 }, load: { FakeModel(name: "x", bytes: 0, ledger: MemoryLedger(capacity: 0)) })
            _ = try await touch(manager, .embedder)
        }
    }

    @Test func theSystemProbeReportsPlausibleMemory() {
        let available = SystemMemoryProbe().availableBytes()
        #expect(available > 0 && available <= ProcessInfo.processInfo.physicalMemory)
    }
}

@Suite("Memory estimates")
struct MemoryEstimateTests {
    private func header(arch: String, layers: Int? = 28, heads: Int? = 16, kvHeads: Int? = 8, embedding: Int = 1024,
                        keyLength: Int? = 128, size: UInt64 = 1_000_000) -> GGUFHeader {
        GGUFHeader(
            name: "m", architecture: arch, embeddingLength: embedding, hasClassifierHead: false, bf16TensorNames: [], poolingType: nil, hasChatTemplate: true, blockCount: layers,
            headCount: heads, headCountKV: kvHeads, keyLength: keyLength, valueLength: keyLength, fileSize: size)
    }

    @Test func decoderModelsCountTheKVCache() {
        // 28 layers × 1024 tokens × 8 KV heads × (128 + 128) × 2 bytes
        let estimate = MemoryEstimate.estimate(header: header(arch: "qwen3"), contextTokens: 1024)
        #expect(estimate.weights == 1_000_000)
        #expect(estimate.kvCache == 28 * 1024 * 8 * 256 * 2)
        #expect(estimate.total == estimate.weights + estimate.kvCache + estimate.compute)
    }

    @Test func encoderOnlyModelsHaveNoKVCache() {
        for arch in ["bert", "nomic-bert", "xlm-roberta", "modern-bert"] {
            #expect(MemoryEstimate.estimate(header: header(arch: arch), contextTokens: 2048).kvCache == 0, "\(arch)")
        }
    }

    @Test func kvCacheGrowsWithContextAndShrinksWithGroupedQueryAttention() {
        let small = MemoryEstimate.estimate(header: header(arch: "llama"), contextTokens: 1024).kvCache
        let large = MemoryEstimate.estimate(header: header(arch: "llama"), contextTokens: 4096).kvCache
        #expect(large == small * 4)
        let multiHead = MemoryEstimate.estimate(header: header(arch: "llama", kvHeads: 16), contextTokens: 1024).kvCache
        #expect(multiHead == small * 2)
    }

    @Test func missingMetadataFallsBackToConservativeValuesWithoutCrashing() {
        let bare = GGUFHeader(
            name: nil, architecture: nil, embeddingLength: nil, hasClassifierHead: false, bf16TensorNames: [], poolingType: nil, hasChatTemplate: true, blockCount: nil, headCount: nil,
            headCountKV: nil, keyLength: nil, valueLength: nil, fileSize: 500)
        let estimate = MemoryEstimate.estimate(header: bare, contextTokens: 0)
        #expect(estimate.weights == 500 && estimate.total > 500)
    }

    @Test func headDimensionDefaultsToEmbeddingOverHeads() {
        let implicit = MemoryEstimate.estimate(header: header(arch: "llama", embedding: 2048, keyLength: nil), contextTokens: 100).kvCache
        #expect(implicit == 28 * 100 * 8 * ((2048 / 16) * 2) * 2)
    }

    @Test func estimatingAMissingFileThrows() {
        #expect(throws: LlamaError.fileNotFound) { try MemoryEstimate.estimate(modelAt: "/no/such/model.gguf", contextTokens: 512) }
    }
}
