import Foundation

/// How well the cited sources support a sentence.
public enum SupportLabel: String, Sendable, Equatable {
    case supported, weak, unsupported
}

/// Score cut-offs on a 0–1 support score.
public struct SupportThresholds: Sendable, Equatable {
    /// At or above this a sentence is `.supported`.
    public var supported: Float
    /// At or above this (and below `supported`) it is `.weak`; below it, `.unsupported`.
    public var weak: Float

    public init(supported: Float, weak: Float) {
        self.supported = max(supported, weak)  // keep the bands ordered
        self.weak = weak
    }

    public func label(for score: Float) -> SupportLabel {
        score >= supported ? .supported : score >= weak ? .weak : .unsupported
    }
}

/// Scores how well each passage supports a claim, 0 to 1. Called once per sentence with its candidate passages.
public typealias SupportScorer = @Sendable (_ claim: String, _ passages: [String]) async throws -> [Float]

/// One sentence of a verified answer.
public struct VerifiedSentence: Sendable, Equatable {
    /// The sentence as it appears in the final answer text: invalid markers removed, the rest renumbered, a recovered
    /// citation added.
    public var text: String
    /// Source numbers as the model saw them (1-based), including a recovered one.
    public var citedSources: [Int]
    /// The same citations in the final numbering (order of first use across the answer).
    public var citationIndices: [Int]
    public var label: SupportLabel
    /// Highest support score among the cited sources (or, for an uncited sentence, among all sources).
    public var score: Float
    /// True when the model gave no citation and one was attached because a source supported the sentence.
    public var recovered: Bool
}

public struct VerificationResult: Sendable {
    /// The answer text with citations resolved and numbered by first use.
    public var text: String
    public var sentences: [VerifiedSentence]
    /// Source numbers (1-based, as the model saw them) in order of first use; `usedSources[i]` is cited as `[i + 1]`.
    public var usedSources: [Int]
    /// Supported sentences divided by all sentences (0 for an empty answer).
    public var supportedFraction: Double
    /// The worst label among the sentences.
    public var overall: SupportLabel
    /// True when more than half of the sentences are unsupported (or there are none), the condition for refusing.
    public var mostlyUnsupported: Bool
}

/// Checks every sentence of a generated answer against the sources it cites.
public enum AnswerVerifier {
    /// Verifies `raw`, the model's reply, against `sources` (the passages in the order they were numbered in the prompt).
    ///
    /// For each sentence:
    /// 1. Markers naming a source that does not exist are removed. A sentence whose markers were all invalid is
    ///    `.unsupported`: the model cited something that is not there.
    /// 2. A cited sentence is scored against each cited source; its score is the best of them.
    /// 3. A sentence with no markers is scored against every source. If the best one reaches the `supported`
    ///    threshold its citation is attached ("recovered"); otherwise the sentence is `.unsupported`, since it
    ///    carries no citation.
    /// 4. The score is mapped to supported, weak or unsupported by `thresholds`.
    ///
    /// Citations are renumbered in order of first use. `onSentence` is called as each sentence is finished, with
    /// its final numbering, which is known as soon as the sentences before it are.
    public static func verify(
        _ raw: String, sources: [String], thresholds: SupportThresholds, scorer: SupportScorer,
        onSentence: (@Sendable (VerifiedSentence) -> Void)? = nil
    ) async throws -> VerificationResult {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        var numbering: [Int: Int] = [:]  // model's source number → final number
        var used: [Int] = []
        func finalNumber(_ source: Int) -> Int {
            if let known = numbering[source] { return known }
            used.append(source)
            numbering[source] = used.count
            return used.count
        }

        var rebuilt = ""
        var cursor = text.startIndex
        var sentences: [VerifiedSentence] = []

        for sentence in SentenceSplitter.split(text) {
            try Task.checkCancellation()
            rebuilt += text[cursor..<sentence.range.lowerBound]
            cursor = sentence.range.upperBound

            let original = String(text[sentence.range])
            let markers = CitationParser.markers(in: original)
            var cited: [Int] = []
            var sawInvalid = false
            for number in markers.flatMap(\.numbers) {
                if (1...max(sources.count, 1)).contains(number), !sources.isEmpty {
                    if !cited.contains(number) { cited.append(number) }
                } else {
                    sawInvalid = true
                }
            }
            let claim = SentenceSplitter.body(of: sentence.range, in: text)

            var score: Float = 0
            var recovered = false
            var label = SupportLabel.unsupported
            if !cited.isEmpty {
                let scores = try await scorer(claim, cited.map { sources[$0 - 1] })
                score = scores.max() ?? 0
                label = thresholds.label(for: score)
            } else if !sawInvalid, !sources.isEmpty {
                let scores = try await scorer(claim, sources)
                if let best = scores.enumerated().max(by: { $0.element < $1.element }) {
                    score = best.element
                    if score >= thresholds.supported {
                        cited = [best.offset + 1]
                        recovered = true
                        label = .supported
                    }
                }
            }

            for source in cited { _ = finalNumber(source) }
            let indices = cited.compactMap { numbering[$0] }
            let finalText = rewrite(original, markers: markers, numbering: numbering, sourceCount: sources.count, recovered: recovered ? indices : [])
            rebuilt += finalText

            let verified = VerifiedSentence(
                text: finalText, citedSources: cited, citationIndices: indices, label: label, score: score, recovered: recovered)
            sentences.append(verified)
            onSentence?(verified)
        }
        rebuilt += text[cursor...]

        let supported = sentences.filter { $0.label == .supported }.count
        let unsupported = sentences.filter { $0.label == .unsupported }.count
        let overall: SupportLabel = unsupported > 0 ? .unsupported : sentences.contains { $0.label == .weak } ? .weak : .supported
        return VerificationResult(
            text: rebuilt.trimmingCharacters(in: .whitespacesAndNewlines), sentences: sentences, usedSources: used,
            supportedFraction: sentences.isEmpty ? 0 : Double(supported) / Double(sentences.count), overall: overall,
            mostlyUnsupported: sentences.isEmpty || unsupported * 2 > sentences.count)
    }

    /// The sentence with markers rewritten to the final numbering, invalid ones dropped and, when a citation was
    /// recovered, `[n]` placed before the closing punctuation.
    private static func rewrite(
        _ sentence: String, markers: [CitationParser.Marker], numbering: [Int: Int], sourceCount: Int, recovered: [Int]
    ) -> String {
        var output = ""
        var cursor = sentence.startIndex
        var removed = false
        for marker in markers {
            output += sentence[cursor..<marker.range.lowerBound]
            cursor = marker.range.upperBound
            let valid = marker.numbers.compactMap { number -> String? in
                guard let final = numbering[number], number >= 1, number <= sourceCount else { return nil }
                return "[\(final)]"
            }
            if valid.isEmpty { removed = true }
            output += valid.joined()
        }
        output += sentence[cursor...]
        if removed { output = CitationParser.tidy(output) }

        guard !recovered.isEmpty else { return output }
        let tags = recovered.map { "[\($0)]" }.joined()
        var end = output.endIndex
        while end > output.startIndex, ".!?…\"”’)".contains(output[output.index(before: end)]) { end = output.index(before: end) }
        return String(output[..<end]) + " " + tags + String(output[end...])
    }
}
