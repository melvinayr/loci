import Foundation
import llama
import os

private let log = Logger(subsystem: "dev.loci", category: "llama")

/// Process-wide llama.cpp backend. Initialised once on first use and kept for the
/// lifetime of the process, so there is nothing to free.
enum LlamaBackend {
    /// Forces backend initialisation. Idempotent and thread-safe.
    static func ensureInitialized() {
        _ = initialized
    }

    /// Static initialisers run exactly once, lazily, with a lock.
    private static let initialized: Bool = {
        // llama.cpp is chatty on stderr; keep errors only, routed to the unified log as private.
        // `LOCI_LLAMA_LOG=1` prints everything to stderr, for debugging a model that will not load or run.
        if ProcessInfo.processInfo.environment["LOCI_LLAMA_LOG"] == nil {
            llama_log_set({ level, text, _ in
                guard level == GGML_LOG_LEVEL_ERROR, let text else { return }
                log.error("\(String(cString: text), privacy: .private)")
            }, nil)
        }
        llama_backend_init()
        return true
    }()

    /// Compiled-in CPU/GPU features of the linked build, for diagnostics.
    static var systemInfo: String {
        ensureInitialized()
        return String(cString: llama_print_system_info())
    }
}
