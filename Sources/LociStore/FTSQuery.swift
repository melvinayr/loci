/// Turns free text into a safe FTS5 `MATCH` expression.
enum FTSQuery {
    static let maxTerms = 64

    /// Distinct lowercase alphanumeric terms, each quoted, joined with `OR` so BM25
    /// ranks chunks matching more (and rarer) terms higher. Nil when no terms remain.
    static func build(_ text: String) -> String? {
        var terms: [String] = []
        var seen = Set<String>()
        var current = ""
        func commit() {
            if !current.isEmpty, seen.insert(current).inserted, terms.count < maxTerms { terms.append(current) }
            current = ""
        }
        for character in text.lowercased() {
            if character.isLetter || character.isNumber {
                current.append(character)
            } else {
                commit()
            }
        }
        commit()
        guard !terms.isEmpty else { return nil }
        return terms.map { "\"\($0)\"" }.joined(separator: " OR ")
    }
}
