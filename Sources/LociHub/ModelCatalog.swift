import Foundation
import Loci

/// One model file Loci can use, with what a user needs to know before downloading it.
public struct CatalogModel: Sendable, Hashable, Identifiable {
    public enum Role: String, Sendable, Hashable {
        case embedder, reranker, generator
    }

    public var id: String { "\(source.repo ?? "")/\(source.file)" }
    public let role: Role
    public let name: String
    public let source: ModelSource
    /// Size of the file on disk.
    public let sizeBytes: Int64
    /// SPDX identifier of the licence, as reported by the model's repository.
    public let license: String
    public let notes: String

    /// Memory while loaded: the file's size plus a fifth for the context and buffers. An estimate, not a measurement.
    public var estimatedMemoryBytes: Int64 { sizeBytes + sizeBytes / 5 }

}

/// A named `ModelSet` with its models described.
public struct CatalogSet: Sendable, Hashable, Identifiable {
    public var id: String { name }
    public let name: String
    public let summary: String
    public let set: ModelSet
    public let models: [CatalogModel]

    public var totalSizeBytes: Int64 { models.reduce(0) { $0 + $1.sizeBytes } }
    /// Peak memory in `.lowMemory` residency, where only the largest model is loaded at a time, plus a fifth for buffers.
    public var lowMemoryEstimateBytes: Int64 { models.map(\.estimatedMemoryBytes).max() ?? 0 }
}

/// The curated model sets, matching `docs/MODELS.md`. The measurements behind the choices are in `docs/reports`.
public enum ModelCatalog {
    public static let all: [CatalogSet] = [recommended, quality]

    public static let recommended = CatalogSet(
        name: "Recommended",
        summary: "English documents on iPhone or Mac.",
        set: .recommended,
        models: [bgeSmall, qwen3Reranker, qwen3Generator])

    public static let quality = CatalogSet(
        name: "Quality",
        summary: "The same reranker and generator with a larger embedder, for Macs and Pro iPhones.",
        set: .quality,
        models: [nomic, qwen3Reranker, qwen3Generator])

    static let bgeSmall = model(
        .embedder, "bge-small-en-v1.5 Q8_0", ModelSet.recommended.embedder, 36_685_152, "MIT", "English only. 384 dimensions.")
    static let nomic = model(
        .embedder, "nomic-embed-text-v1.5 Q8_0", ModelSet.quality.embedder, 146_146_432, "Apache-2.0", "English. 768 dimensions.")
    static let qwen3Reranker = model(
        .reranker, "Qwen3-Reranker-0.6B Q8_0", reranker, 639_153_184, "Apache-2.0",
        "Also verifies each answer sentence against its sources.")
    static let qwen3Generator = model(
        .generator, "Qwen3-4B Q4_K_M", generator, 2_497_280_256, "Apache-2.0", "Thinking is disabled; answers cite passages as [n].")

    private static var reranker: ModelSource { ModelSet.recommended.reranker ?? ModelSet.recommended.embedder }
    private static var generator: ModelSource { ModelSet.recommended.generator ?? ModelSet.recommended.embedder }

    private static func model(
        _ role: CatalogModel.Role, _ name: String, _ source: ModelSource, _ size: Int64, _ license: String, _ notes: String
    ) -> CatalogModel {
        CatalogModel(role: role, name: name, source: source, sizeBytes: size, license: license, notes: notes)
    }
}

extension ModelSource {
    fileprivate var repo: String? {
        if case .hub(let repo, _, _) = self { repo } else { nil }
    }
    fileprivate var file: String {
        if case .hub(_, let file, _) = self { file } else { "" }
    }
}

extension ModelDownloader {
    /// Downloads every model of `set` that is missing, one after another, reporting progress per model. Returns when all are present and verified.
    public func ensure(
        _ set: CatalogSet, progress: (@Sendable (_ model: CatalogModel, _ progress: DownloadProgress) -> Void)? = nil
    ) async throws {
        for model in set.models {
            try Task.checkCancellation()
            _ = try await ensure(model.source) { progress?(model, $0) }
        }
    }
}
