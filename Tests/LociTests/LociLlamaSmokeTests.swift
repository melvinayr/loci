import Testing
@testable import LociLlama

@Suite("llama.cpp backend")
struct LlamaBackendTests {
    @Test func backendInitialisesRepeatedly() {
        LlamaBackend.ensureInitialized()
        LlamaBackend.ensureInitialized()
        #expect(!LlamaBackend.systemInfo.isEmpty)
    }
}
