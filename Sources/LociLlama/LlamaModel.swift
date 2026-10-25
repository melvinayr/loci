import Foundation
import llama
import os

/// Owns one `llama_model *` and frees it in `deinit`.
///
/// The model is immutable after loading, so tokenizing from any thread is safe (llama.cpp
/// treats the vocabulary as read-only); only contexts are confined to `LlamaExecutor`.
public final class LlamaModel: @unchecked Sendable {
    // Safety of @unchecked Sendable: `pointer` is set once in init and never mutated, every member
    // below only reads from the model (llama.cpp treats it as read-only after load), and the one
    // cache, `specialTokens`, is guarded by a lock.
    let pointer: OpaquePointer
    public let fileSize: UInt64

    /// - Parameters:
    ///   - gpuLayers: 0 for CPU only, a large number to offload everything.
    init(path: String, gpuLayers: Int32) throws {
        LlamaBackend.ensureInitialized()
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
              let size = (attributes[.size] as? NSNumber)?.uint64Value else {
            throw LlamaError.fileNotFound
        }
        var params = llama_model_default_params()
        params.n_gpu_layers = gpuLayers

        // With GPU offload, keep BF16 weights on the CPU: the Metal library in the pinned llama.cpp
        // build lacks the BF16 kernel and aborts the whole process if it is asked to multiply them.
        let pinned = gpuLayers == 0 ? [] : ((try? GGUFHeader.read(path: path).bf16TensorNames) ?? [])
        var patterns: [UnsafeMutablePointer<CChar>] = []
        defer { patterns.forEach { free($0) } }
        var overrides: [llama_model_tensor_buft_override] = []
        for name in pinned {
            guard let pattern = strdup("^" + NSRegularExpression.escapedPattern(for: name) + "$") else { continue }
            patterns.append(pattern)
            overrides.append(llama_model_tensor_buft_override(pattern: UnsafePointer(pattern), buft: ggml_backend_cpu_buffer_type()))
        }
        overrides.append(llama_model_tensor_buft_override(pattern: nil, buft: nil))  // terminator

        let loaded: OpaquePointer? = overrides.withUnsafeBufferPointer { buffer in
            params.tensor_buft_overrides = pinned.isEmpty ? nil : buffer.baseAddress
            return llama_model_load_from_file(path, params)
        }
        guard let loaded else {
            throw LlamaError.invalidModel(reason: "llama.cpp could not load the file as a GGUF model")
        }
        pointer = loaded
        fileSize = size
    }

    deinit {
        llama_model_free(pointer)
    }

    private var vocab: OpaquePointer? { llama_model_get_vocab(pointer) }

    /// Output embedding dimension.
    var embeddingDimension: Int { Int(llama_model_n_embd_out(pointer)) }
    var trainingContext: Int { Int(llama_model_n_ctx_train(pointer)) }
    var hasEncoder: Bool { llama_model_has_encoder(pointer) }

    func metadata(_ key: String) -> String? {
        var buffer = [CChar](repeating: 0, count: 512)
        let length = llama_model_meta_val_str(pointer, key, &buffer, buffer.count)
        guard length >= 0 else { return nil }
        return String(decoding: buffer.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    var name: String { metadata("general.name") ?? "" }
    var architecture: String { metadata("general.architecture") ?? "" }

    /// Tokens for `text`. `parseSpecial` lets chat markers such as `<|im_start|>` become special tokens;
    /// leave it off for any text that did not come from the model's own template.
    func tokenize(_ text: String, addSpecial: Bool, parseSpecial: Bool = false) -> [llama_token] {
        guard let vocab else { return [] }
        let bytes = Array(text.utf8)
        return bytes.withUnsafeBufferPointer { buffer in
            let base = UnsafeRawPointer(buffer.baseAddress).map { $0.assumingMemoryBound(to: CChar.self) }
            var capacity = max(bytes.count / 2, 16) + 8
            while true {
                var tokens = [llama_token](repeating: 0, count: capacity)
                let count = llama_tokenize(vocab, base, Int32(bytes.count), &tokens, Int32(capacity), addSpecial, parseSpecial)
                if count >= 0 { return Array(tokens.prefix(Int(count))) }
                capacity = Int(-count)
            }
        }
    }

    /// Text for `tokens`, without special tokens, going through a UTF-8 buffer.
    func detokenize(_ tokens: [llama_token]) -> String {
        guard let vocab, !tokens.isEmpty else { return "" }
        var capacity = tokens.count * 8 + 16
        while true {
            var buffer = [CChar](repeating: 0, count: capacity)
            let count = llama_detokenize(vocab, tokens, Int32(tokens.count), &buffer, Int32(capacity), true, false)
            if count >= 0 {
                return String(decoding: buffer.prefix(Int(count)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
            }
            capacity = Int(-count) + 1
        }
    }

    /// The literal text of every control token (`<s>`, `<|im_end|>`, …), longest first. Removing these
    /// from untrusted text before a template is tokenized with special parsing stops a document
    /// from injecting chat markers.
    var specialTokenTexts: [String] {
        specialTokens.withLock { cached in
            if let cached { return cached }
            var texts = Set<String>()
            if let vocab {
                for token in 0..<llama_vocab_n_tokens(vocab) where llama_vocab_is_control(vocab, token) {
                    if let raw = llama_vocab_get_text(vocab, token) {
                        let text = String(cString: raw)
                        if text.count >= 3 { texts.insert(text) }
                    }
                }
            }
            let sorted = texts.sorted { $0.count > $1.count }
            cached = sorted
            return sorted
        }
    }

    private let specialTokens = OSAllocatedUnfairLock<[String]?>(initialState: nil)

    /// `text` with every special-token string removed.
    func removingSpecialTokenTexts(_ text: String) -> String {
        var result = text
        for special in specialTokenTexts where result.contains(special) {
            result = result.replacingOccurrences(of: special, with: "")
        }
        return result
    }

    /// Number of classifier outputs (rerankers); 0 for models without a classification head.
    var classifierOutputCount: Int { Int(llama_model_n_cls_out(pointer)) }

    func classifierLabel(_ index: Int) -> String? {
        llama_model_cls_label(pointer, UInt32(index)).map { String(cString: $0) }
    }

    /// The embedded chat template; `name` selects a named variant such as `"rerank"`, nil the default.
    func chatTemplate(named name: String?) -> String? {
        llama_model_chat_template(pointer, name).map { String(cString: $0) }
    }

    /// The text of the beginning-of-sequence token, empty when the model has none.
    var beginText: String {
        guard let vocab, beginToken >= 0, let raw = llama_vocab_get_text(vocab, beginToken) else { return "" }
        return String(cString: raw)
    }

    /// True when the tokenizer adds a BOS token to the start of a prompt on its own.
    var addsBeginToken: Bool { vocab.map { llama_vocab_get_add_bos($0) } ?? false }

    func isEndOfGeneration(_ token: llama_token) -> Bool {
        vocab.map { llama_vocab_is_eog($0, token) } ?? true
    }

    /// The raw UTF-8 bytes of one token, with special tokens rendered as their text so reasoning
    /// markers can be recognised. A piece may end in the middle of a character.
    func pieceBytes(_ token: llama_token) -> [UInt8] {
        guard let vocab else { return [] }
        var capacity = 64
        while true {
            var buffer = [CChar](repeating: 0, count: capacity)
            let count = llama_token_to_piece(vocab, token, &buffer, Int32(capacity), 0, true)
            if count >= 0 { return buffer.prefix(Int(count)).map { UInt8(bitPattern: $0) } }
            capacity = Int(-count) + 1
        }
    }

    var beginToken: llama_token { llama_vocab_bos(vocab) }
    var endToken: llama_token { llama_vocab_eos(vocab) }
    var separatorToken: llama_token { llama_vocab_sep(vocab) }

    /// Number of tokens in `scalars` without special tokens; used to size chunks.
    func tokenCount(_ scalars: ArraySlice<Unicode.Scalar>) -> Int {
        guard !scalars.isEmpty, let vocab else { return 0 }
        let bytes = Array(String(String.UnicodeScalarView(scalars)).utf8)
        return bytes.withUnsafeBufferPointer { buffer in
            let base = UnsafeRawPointer(buffer.baseAddress).map { $0.assumingMemoryBound(to: CChar.self) }
            // With no room for output, llama_tokenize returns the negated token count.
            let result = llama_tokenize(vocab, base, Int32(bytes.count), nil, 0, false, false)
            return Int(result < 0 ? -result : result)
        }
    }
}
