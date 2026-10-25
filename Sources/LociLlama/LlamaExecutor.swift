import Dispatch
import os

/// The one serial queue all llama.cpp work runs on, off the cooperative thread pool.
/// Serialising across models also keeps peak memory and GPU contention predictable.
enum LlamaExecutor {
    static let queue = DispatchQueue(label: "dev.loci.llama", qos: .userInitiated)

    /// Runs `work` on the queue. `isCancelled` turns true when the calling task is cancelled so
    /// long loops can stop between batches; a task cancelled before the work starts never runs it.
    static func run<T: Sendable>(
        _ work: @escaping @Sendable (_ isCancelled: @Sendable () -> Bool) throws -> T
    ) async throws -> T {
        let flag = CancellationFlag()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    if flag.isSet {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    do {
                        continuation.resume(returning: try work { flag.isSet })
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        } onCancel: {
            flag.set()
        }
    }
}

final class CancellationFlag: Sendable {
    private let state = OSAllocatedUnfairLock(initialState: false)
    var isSet: Bool { state.withLock { $0 } }
    func set() { state.withLock { $0 = true } }
}
