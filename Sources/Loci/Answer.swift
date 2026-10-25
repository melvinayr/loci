import Foundation
internal import LociGrounding

/// How well the cited passages support a sentence of an answer.
public enum SupportLevel: String, Sendable, Codable {
    case supported, weak, unsupported
}

extension SupportLevel {
    init(_ label: SupportLabel) {
        switch label {
        case .supported: self = .supported
        case .weak: self = .weak
        case .unsupported: self = .unsupported
        }
    }
}

/// A passage an answer cites.
public struct Citation: Sendable, Hashable {
    /// The `[n]` shown to the user.
    public let index: Int
    public let document: Document
    public let chunkID: ChunkID
    /// 1-based, PDFs only; always contains `range`.
    public let page: Int?
    /// Unicode-scalar offsets in the document's normalised text.
    public let range: Range<Int>
    /// The passage text, exactly the slice of the document text at `range`.
    public let excerpt: String
    /// The last score that ranked the passage: the reranker's when it ran, else the fused,
    /// vector or keyword score. Scales differ between these.
    public let retrievalScore: Float
}

/// One sentence of an answer with its citations and how well they support it.
public struct AnsweredSentence: Sendable {
    public let text: String
    public let citationIndices: [Int]
    public let support: SupportLevel
    /// 0–1 from the support scorer.
    public let score: Float
}

/// Overall grounding of an answer.
public struct SupportSummary: Sendable {
    /// Supported sentences divided by all sentences.
    public let supportedFraction: Double
    /// The worst level among the sentences.
    public let level: SupportLevel
}

/// How long each stage took and what it cost.
public struct AnswerStats: Sendable {
    /// Embedding the question, searching and fusing. Excludes reranking.
    public let retrievalTime: Duration
    /// The reranker, including loading it if it was not resident.
    public let rerankTime: Duration
    /// From the start of generation, including loading the generator and reading the prompt, to the first visible text.
    public let timeToFirstToken: Duration
    public let generationTokensPerSecond: Double
    /// Scoring every sentence against its sources; zero when verification did not run.
    public let verificationTime: Duration
    /// Tokens in the prompt: instructions, numbered sources and question.
    public let contextTokens: Int
    /// The highest memory footprint the process has reached so far, which can predate this answer.
    public let peakMemoryBytes: UInt64
}

/// A grounded answer.
public struct Answer: Sendable {
    /// The reply with inline `[n]` markers. Markers naming sources that do not exist are removed, and
    /// the rest are numbered in order of first use, so `[1]` is always `citations[0]`.
    public let text: String
    /// The answer's sentences, each with its citations and how well they support it. Empty when `support` is nil.
    public let sentences: [AnsweredSentence]
    /// The cited passages, numbered in order of first use. When `refused`, the passages that were retrieved instead, in
    /// rank order, so the user can see what was looked at.
    public let citations: [Citation]
    /// How well the sources support the answer. Nil when verification did not run (`verify: false`) and for a refusal.
    public let support: SupportSummary?
    /// True when the documents do not contain an answer: the model said so, nothing relevant was found, or most of its
    /// answer was not supported by the sources it cited (with `refuseWhenUnsupported`). `text` is then the fixed refusal sentence.
    public let refused: Bool
    public let stats: AnswerStats
}

/// Progress of `streamAnswer`.
public enum AnswerEvent: Sendable {
    /// The passages the answer will draw on, before generation starts, so a UI can show sources early.
    case retrieved([SearchResult])
    /// The next piece of the reply as generated: never a partial character and without reasoning blocks,
    /// but with the model's own citation numbers. `Answer.text` has them cleaned and renumbered.
    case token(String)
    /// A sentence that has just been verified, with its final citation numbers. These arrive after generation finishes and before
    /// `.finished`; if most sentences turn out unsupported and the answer is refused, `.finished` is authoritative.
    case sentence(AnsweredSentence)
    case finished(Answer)
}

/// Settings for `KnowledgeBase.ask` and `streamAnswer`.
public struct AnswerOptions: Sendable {
    public var retrieval: RetrievalOptions
    /// Cap on prompt tokens (instructions, sources and question); nil uses the generator's context minus `maxAnswerTokens`.
    public var maxContextTokens: Int?
    public var maxAnswerTokens: Int
    /// Low by default: grounded answers should be conservative.
    public var temperature: Float
    /// Check every sentence against the sources it cites (and attach a citation to an uncited sentence a source supports).
    public var verify: Bool
    /// Support-score cut-offs for `.supported` and `.weak`. The score scale depends on the reranker (or the embedder, without one), so
    /// calibrate them for your model set with `loci-cli eval --calibrate`; `docs/MODELS.md` lists calibrated values.
    public var supportThresholds: (supported: Float, weak: Float)
    /// Refuse, instead of presenting the answer, when more than half of its sentences are unsupported. The grounding prompt
    /// already tells the model to refuse when the sources lack the answer; this catches answers it made up anyway.
    public var refuseWhenUnsupported: Bool
    /// Appended to the grounding system prompt.
    public var instructions: String?

    public init(
        retrieval: RetrievalOptions = .default,
        maxContextTokens: Int? = nil,
        maxAnswerTokens: Int = 400,
        temperature: Float = 0.2,
        verify: Bool = true,
        supportThresholds: (supported: Float, weak: Float) = (0.6, 0.35),
        refuseWhenUnsupported: Bool = true,
        instructions: String? = nil
    ) {
        self.retrieval = retrieval
        self.maxContextTokens = maxContextTokens
        self.maxAnswerTokens = maxAnswerTokens
        self.temperature = temperature
        self.verify = verify
        self.supportThresholds = supportThresholds
        self.refuseWhenUnsupported = refuseWhenUnsupported
        self.instructions = instructions
    }

    public static let `default` = AnswerOptions()
}
