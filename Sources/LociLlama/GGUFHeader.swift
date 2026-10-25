import Foundation
import llama

/// The few metadata keys Loci needs, read straight from a GGUF file's header without
/// loading any weights.
struct GGUFHeader {
    var name: String?
    var architecture: String?
    var embeddingLength: Int?
    /// True when the file contains a classification head (`cls.*` tensors): the mark of a reranker.
    var hasClassifierHead: Bool
    /// Tensors stored as BF16. The prebuilt Metal library has no BF16 matrix-multiply kernel, so these must run on the CPU.
    var bf16TensorNames: [String]
    /// `<arch>.pooling_type`: 1 mean, 2 CLS, 3 last, 4 rank; nil when the file does not state one.
    var poolingType: Int?
    var hasChatTemplate: Bool
    var blockCount: Int?
    var headCount: Int?
    var headCountKV: Int?
    /// Per-head key and value sizes when the file states them (otherwise embedding length ÷ head count).
    var keyLength: Int?
    var valueLength: Int?
    var fileSize: UInt64

    static func read(path: String) throws -> GGUFHeader {
        LlamaBackend.ensureInitialized()
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
              let size = (attributes[.size] as? NSNumber)?.uint64Value else {
            throw LlamaError.fileNotFound
        }
        guard let context = gguf_init_from_file(path, gguf_init_params(no_alloc: true, ctx: nil)) else {
            throw LlamaError.invalidModel(reason: "the file is not a GGUF model")
        }
        defer { gguf_free(context) }

        func string(_ key: String) -> String? {
            let id = gguf_find_key(context, key)
            guard id >= 0, gguf_get_kv_type(context, id) == GGUF_TYPE_STRING, let value = gguf_get_val_str(context, id) else {
                return nil
            }
            return String(decoding: Data(bytes: value, count: strlen(value)), as: UTF8.self)
        }
        func integer(_ key: String) -> Int? {
            let id = gguf_find_key(context, key)
            guard id >= 0 else { return nil }
            switch gguf_get_kv_type(context, id) {
            case GGUF_TYPE_UINT32: return Int(gguf_get_val_u32(context, id))
            case GGUF_TYPE_INT32: return Int(gguf_get_val_i32(context, id))
            case GGUF_TYPE_UINT64: return Int(exactly: gguf_get_val_u64(context, id))
            default: return nil
            }
        }

        var hasClassifierHead = false
        var bf16Tensors: [String] = []
        for index in 0..<gguf_get_n_tensors(context) {
            guard let raw = gguf_get_tensor_name(context, index) else { continue }
            let name = String(cString: raw)
            if name.hasPrefix("cls.") { hasClassifierHead = true }
            if gguf_get_tensor_type(context, index) == GGML_TYPE_BF16 { bf16Tensors.append(name) }
        }

        let architecture = string("general.architecture")
        return GGUFHeader(
            name: string("general.name"), architecture: architecture,
            embeddingLength: architecture.flatMap { integer("\($0).embedding_length") },
            hasClassifierHead: hasClassifierHead, bf16TensorNames: bf16Tensors,
            poolingType: architecture.flatMap { integer("\($0).pooling_type") },
            hasChatTemplate: gguf_find_key(context, "tokenizer.chat_template") >= 0,
            blockCount: architecture.flatMap { integer("\($0).block_count") },
            headCount: architecture.flatMap { integer("\($0).attention.head_count") },
            headCountKV: architecture.flatMap { integer("\($0).attention.head_count_kv") },
            keyLength: architecture.flatMap { integer("\($0).attention.key_length") },
            valueLength: architecture.flatMap { integer("\($0).attention.value_length") },
            fileSize: size)
    }
}

extension GGUFHeader {
    /// A stable identity built only from header fields: `<name>|<embedding_length>|<file size>`.
    func identityString() -> String {
        let label = name?.isEmpty == false ? name ?? "unnamed" : "unnamed"
        return "\(label)|\(embeddingLength ?? 0)|\(fileSize)"
    }
}

extension GGUFHeader {
    /// Embedders must pool a sequence into one vector (mean, CLS or last token), which the file states.
    func requireEmbedderRole() throws {
        guard let poolingType, (1...3).contains(poolingType), embeddingLength != nil else {
            throw LlamaError.roleMismatch(expected: "embedding")
        }
    }

    func requireRerankerRole() throws {
        guard hasClassifierHead else { throw LlamaError.roleMismatch(expected: "reranker") }
    }

    func requireGeneratorRole() throws {
        guard !hasClassifierHead, !MemoryEstimate.encoderOnly.contains(architecture ?? ""), hasChatTemplate else {
            throw LlamaError.roleMismatch(expected: "generator")
        }
    }
}
