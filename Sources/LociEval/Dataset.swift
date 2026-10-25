import Foundation

/// A ground-truth passage a question should be answered from.
public struct RelevantPassage: Codable, Sendable, Equatable {
    /// File name of the document, for example `trial-2024.pdf`.
    public var document: String
    /// 1-based page; when set, a retrieved passage must come from this page.
    public var page: Int?
    /// Text that must appear in the retrieved passage (compared ignoring case and whitespace runs).
    public var snippet: String?

    public init(document: String, page: Int? = nil, snippet: String? = nil) {
        self.document = document
        self.page = page
        self.snippet = snippet
    }
}

/// One line of `questions.jsonl`.
public struct EvalQuestion: Codable, Sendable, Equatable {
    public var id: String
    /// The question as a user would ask it.
    public var question: String
    /// Passages that answer the question; empty for unanswerable questions.
    public var relevant: [RelevantPassage]
    /// Reference answer, for answer-match metrics.
    public var answer: String?
    /// False when the documents do not contain the answer; the right behaviour is refusal.
    public var answerable: Bool

    public init(id: String, question: String, relevant: [RelevantPassage] = [], answer: String? = nil, answerable: Bool = true) {
        self.id = id
        self.question = question
        self.relevant = relevant
        self.answer = answer
        self.answerable = answerable
    }

    enum CodingKeys: String, CodingKey { case id, question, relevant, answer, answerable }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        question = try container.decode(String.self, forKey: .question)
        relevant = try container.decodeIfPresent([RelevantPassage].self, forKey: .relevant) ?? []
        answer = try container.decodeIfPresent(String.self, forKey: .answer)
        answerable = try container.decodeIfPresent(Bool.self, forKey: .answerable) ?? true
    }
}

/// Problems with a dataset or with adding its documents.
public enum EvalError: Error, Sendable, LocalizedError, Equatable {
    case datasetNotFound(String)
    case invalidDataset(String)
    case ingestFailed(document: String, reason: String)
    /// The `hybrid+rerank` mode was requested but the knowledge base has no reranker.
    case rerankerNotConfigured
    /// A section needs something the knowledge base or dataset lacks, such as a generator or a support subset.
    case missing(String)

    public var errorDescription: String? {
        switch self {
        case .datasetNotFound(let what): "Dataset not found: \(what)."
        case .invalidDataset(let reason): "Invalid dataset: \(reason)."
        case .ingestFailed(let document, let reason): "Could not add “\(document)” to the knowledge base: \(reason)"
        case .rerankerNotConfigured: "The hybrid+rerank mode needs a reranker in the knowledge base's ModelSet."
        case .missing(let what): "Cannot evaluate: \(what)."
        }
    }
}

/// Optional `config.json` in a dataset folder: how its documents should be chunked.
public struct DatasetConfig: Codable, Sendable, Equatable {
    /// A `ChunkerKind` raw value: structural, sentence or tokenWindow.
    public var chunker: String?
    /// Target chunk size in tokens; the evaluator falls back to `IngestOptions.default`.
    public var chunkTokens: Int?
    /// Overlap between chunks in tokens.
    public var overlapTokens: Int?

    public init(chunker: String? = nil, chunkTokens: Int? = nil, overlapTokens: Int? = nil) {
        self.chunker = chunker
        self.chunkTokens = chunkTokens
        self.overlapTokens = overlapTokens
    }
}

/// An evaluation dataset.
public enum Dataset: Sendable {
    /// A folder holding `documents/` and `questions.jsonl`. The evaluator adds the documents first.
    case folder(URL)
    /// A `questions.jsonl` file whose documents are already in the knowledge base.
    case jsonl(URL)

    struct Loaded {
        var name: String
        var documentsFolder: URL?
        var questions: [EvalQuestion]
        var config: DatasetConfig
        var supportExamples: [SupportExample]
    }

    func load() throws -> Loaded {
        let manager = FileManager.default
        let questionsURL: URL
        var documents: URL?
        let name: String
        switch self {
        case .folder(let folder):
            questionsURL = folder.appendingPathComponent("questions.jsonl")
            let candidate = folder.appendingPathComponent("documents", isDirectory: true)
            guard manager.fileExists(atPath: candidate.path) else {
                throw EvalError.datasetNotFound("\(folder.lastPathComponent)/documents")
            }
            documents = candidate
            name = folder.lastPathComponent
        case .jsonl(let url):
            questionsURL = url
            name = url.deletingPathExtension().lastPathComponent
        }
        guard let data = try? Data(contentsOf: questionsURL) else {
            throw EvalError.datasetNotFound(questionsURL.lastPathComponent)
        }
        let questions = try Self.parse(data)
        try Self.validate(questions, documentsFolder: documents)

        var config = DatasetConfig()
        if case .folder(let folder) = self,
           let raw = try? Data(contentsOf: folder.appendingPathComponent("config.json")) {
            guard let decoded = try? JSONDecoder().decode(DatasetConfig.self, from: raw) else {
                throw EvalError.invalidDataset("config.json is not valid")
            }
            config = decoded
        }

        var support: [SupportExample] = []
        if case .folder(let folder) = self, let raw = try? Data(contentsOf: folder.appendingPathComponent("support.jsonl")) {
            support = try Self.parseSupport(raw)
            try Self.validate(support, documentsFolder: documents)
        }
        return Loaded(name: name, documentsFolder: documents, questions: questions, config: config, supportExamples: support)
    }

    static func parse(_ data: Data) throws -> [EvalQuestion] {
        guard let text = String(data: data, encoding: .utf8) else {
            throw EvalError.invalidDataset("questions.jsonl is not UTF-8")
        }
        let decoder = JSONDecoder()
        var questions: [EvalQuestion] = []
        for (index, line) in text.split(whereSeparator: \.isNewline).enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }
            do {
                questions.append(try decoder.decode(EvalQuestion.self, from: Data(trimmed.utf8)))
            } catch {
                throw EvalError.invalidDataset("line \(index + 1) of questions.jsonl is not a valid question")
            }
        }
        return questions
    }

    static func parseSupport(_ data: Data) throws -> [SupportExample] {
        guard let text = String(data: data, encoding: .utf8) else { throw EvalError.invalidDataset("support.jsonl is not UTF-8") }
        let decoder = JSONDecoder()
        var examples: [SupportExample] = []
        for (index, line) in text.split(whereSeparator: \.isNewline).enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }
            do {
                examples.append(try decoder.decode(SupportExample.self, from: Data(trimmed.utf8)))
            } catch {
                throw EvalError.invalidDataset("line \(index + 1) of support.jsonl is not a valid example")
            }
        }
        return examples
    }

    static func validate(_ examples: [SupportExample], documentsFolder: URL?) throws {
        var seen = Set<String>()
        for example in examples {
            guard seen.insert(example.id).inserted else { throw EvalError.invalidDataset("duplicate support example id “\(example.id)”") }
            guard !example.claim.trimmingCharacters(in: .whitespaces).isEmpty, !example.passage.trimmingCharacters(in: .whitespaces).isEmpty else {
                throw EvalError.invalidDataset("support example “\(example.id)” has an empty claim or passage")
            }
        }
        if let documentsFolder {
            let names = Set(((try? FileManager.default.subpathsOfDirectory(atPath: documentsFolder.path)) ?? []).map { ($0 as NSString).lastPathComponent })
            for example in examples where !names.contains(example.document) {
                throw EvalError.invalidDataset("support example “\(example.id)” refers to missing document “\(example.document)”")
            }
        }
    }

    static func validate(_ questions: [EvalQuestion], documentsFolder: URL?) throws {
        guard !questions.isEmpty else { throw EvalError.invalidDataset("there are no questions") }
        var seen = Set<String>()
        for question in questions {
            guard seen.insert(question.id).inserted else {
                throw EvalError.invalidDataset("duplicate question id “\(question.id)”")
            }
            if question.answerable && question.relevant.isEmpty {
                throw EvalError.invalidDataset("answerable question “\(question.id)” has no relevant passages")
            }
            if question.relevant.contains(where: { ($0.page ?? 1) < 1 }) {
                throw EvalError.invalidDataset("question “\(question.id)” has a page below 1")
            }
        }
        if let documentsFolder {
            let names = Set(
                ((try? FileManager.default.subpathsOfDirectory(atPath: documentsFolder.path)) ?? []).map {
                    ($0 as NSString).lastPathComponent
                })
            for question in questions {
                for passage in question.relevant where !names.contains(passage.document) {
                    throw EvalError.invalidDataset("question “\(question.id)” refers to missing document “\(passage.document)”")
                }
            }
        }
    }
}
