import Foundation
import Loci
import LociEval
import LociHub

let usage = """
usage: loci-cli <command> [options]

commands:
  index <path>      add a file, or every supported file in a folder
  search <query>    search the knowledge base, printing passages with their page numbers
  ask <question>    answer from the documents with [n] citations (needs --generator)
  documents         list the documents in the knowledge base
  reindex           re-embed every passage after changing the embedding model
  eval              measure retrieval quality on a dataset
  bench             time ingest, search, cold starts and answers on this machine (writes Markdown with --out)
  models list       list downloaded models
  models download <owner/name> <file> [--sha256 <hex>]
                    download a GGUF from Hugging Face into the folder Loci reads models from

options:
  --kb <name>       knowledge base name (default: "default")
  --dir <path>      use an explicit directory instead of Application Support
  --models <set>    embedder: "recommended", "quality", "fixture" (Tests/Fixtures/models), or a .gguf path
  --reranker <m>    reranker: "fixture" (Qwen3-Reranker-0.6B in Tests/Fixtures/models) or a .gguf path
  --generator <m>   generator: "fixture" (Qwen3-0.6B in Tests/Fixtures/models) or a .gguf path
  --residency <p>   lowMemory, balanced or keepResident (default: balanced on macOS)
  --mode <mode>     search: keyword, vector or hybrid (default: hybrid)
  --no-rerank       search/ask: skip the reranker even if one is configured
  --stream          ask: print the reply as it is generated
  --no-verify       ask: skip per-sentence verification
  --instructions <t>  ask: extra instructions appended to the grounding prompt
  --keep-unsupported  ask: show an answer even when most of it is unsupported
  --thresholds <s,w>  ask/eval: support-score cut-offs "supported,weak" (default 0.6,0.35)
  --max-tokens <n>  ask: cap on the reply length (default: 400)
  --temperature <t> ask: sampling temperature (default: 0.2)
  --limit <n>       search/ask: number of passages to retrieve (default: 5)
  --limit <n>       search: number of results (default: 5)

bench options:
  --chunks <n>      passages in the synthetic corpus (default 10000; 50000 matches the spec's target)
  --queries <n>     queries timed per search mode (default 100)
  --rerank-queries <n>   queries timed with reranking (default 10)
  --dataset <dir>   also time full answers on this dataset's questions (needs --generator)
  --cpu             run on the CPU instead of Metal
  --out <file>, --json <file>   write the report

eval options:
  --dataset <dir>           folder with documents/ and questions.jsonl (required)
  --modes <list>            keyword,vector,hybrid,hybrid+rerank or "all" (default: all; hybrid+rerank needs --reranker)
  --out <file>              write a Markdown report
  --json <file>             write a JSON report
  --fail-on-regression <f>  compare with a baseline JSON report; exit 2 if Recall@5 drops
  --margin <points>         allowed Recall@5 drop in percentage points (default: 2)
  --answers                 also evaluate full answers: citations, faithfulness, refusals, answer match, timings (needs --generator)
  --verifier                also compare verification with the hand-labelled support.jsonl
  --calibrate               fit support thresholds on the calibration half of support.jsonl and report both halves
                            (with --answers, the answers use the fitted thresholds)
  --min-precision <p>       with --calibrate: required precision of the "unsupported" flag (default 0.95)
  --no-retrieval            skip the retrieval modes
  --gpu                     use Metal instead of the CPU (needed for practical reranker speed; results
                            can differ slightly from CPU runs)
  --chunk-tokens <n>, --overlap-tokens <n>   override the dataset's chunking
"""

struct Arguments {
    var positional: [String] = []
    var values: [String: String] = [:]

    private static let valueFlags: Set<String> = [
        "--kb", "--dir", "--models", "--reranker", "--generator", "--residency", "--max-tokens", "--temperature", "--thresholds", "--min-precision", "--instructions", "--sha256", "--chunks", "--queries", "--rerank-queries", "--mode", "--limit", "--dataset", "--modes", "--out", "--json",
        "--fail-on-regression", "--margin", "--chunk-tokens", "--overlap-tokens",
    ]

    init?(_ raw: [String]) {
        var iterator = raw.makeIterator()
        while let argument = iterator.next() {
            if ["--no-rerank", "--cpu", "--gpu", "--stream", "--no-verify", "--keep-unsupported", "--answers", "--verifier", "--calibrate", "--no-retrieval"].contains(argument) {
                values[argument] = "true"
            } else if argument.hasPrefix("--") {
                guard Self.valueFlags.contains(argument), let value = iterator.next() else { return nil }
                values[argument] = value
            } else {
                positional.append(argument)
            }
        }
    }

    subscript(flag: String) -> String? { values[flag] }

    func integer(_ flag: String) -> Int? { values[flag].flatMap(Int.init) }
}

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

func fixtureModel(_ name: String) -> URL {
    let url = URL(fileURLWithPath: "Tests/Fixtures/models/\(name)")
    guard FileManager.default.fileExists(atPath: url.path) else {
        fail("The fixture model \(name) is missing. Run ./Scripts/fetch-test-models.sh from the repository root.")
    }
    return url
}

func modelSet(_ arguments: Arguments) -> ModelSet {
    var models: ModelSet
    switch arguments["--models"] ?? "recommended" {
    case "recommended": models = .recommended
    case "quality": models = .quality
    case "fixture": models = ModelSet(embedder: .file(fixtureModel("bge-small-en-v1.5-q8_0.gguf")))
    case let path where path.hasSuffix(".gguf"):
        models = ModelSet(embedder: .file(URL(fileURLWithPath: (path as NSString).expandingTildeInPath)))
    case let other:
        fail("Unknown model set “\(other)”. Use recommended, quality, fixture or a .gguf path.")
    }
    switch arguments["--reranker"] {
    case nil, "none": break
    case "fixture": models.reranker = .file(fixtureModel("qwen3-reranker-0.6b-q8_0.gguf"))
    case let path? where path.hasSuffix(".gguf"):
        models.reranker = .file(URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
    case let other?:
        fail("Unknown reranker “\(other)”. Use fixture or a .gguf path.")
    }
    switch arguments["--generator"] {
    case nil, "none": break
    case "fixture": models.generator = .file(fixtureModel("qwen3-0.6b-q8_0.gguf"))
    case let path? where path.hasSuffix(".gguf"):
        models.generator = .file(URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
    case let other?:
        fail("Unknown generator “\(other)”. Use fixture or a .gguf path.")
    }
    return models
}

func knowledgeBaseOptions(_ arguments: Arguments, gpu: GPULayers = .all) -> KnowledgeBaseOptions {
    var options = KnowledgeBaseOptions(gpuLayers: gpu)
    switch arguments["--residency"] {
    case nil: break
    case "lowMemory": options.residency = .lowMemory
    case "balanced": options.residency = .balanced
    case "keepResident": options.residency = .keepResident
    case let other?: fail("Unknown residency “\(other)”. Use lowMemory, balanced or keepResident.")
    }
    return options
}

func openKnowledgeBase(_ arguments: Arguments) async throws -> KnowledgeBase {
    let models = modelSet(arguments)
    let options = knowledgeBaseOptions(arguments)
    if let directory = arguments["--dir"] {
        return try await KnowledgeBase(directory: URL(fileURLWithPath: directory, isDirectory: true), models: models, options: options)
    }
    return try await KnowledgeBase(name: arguments["--kb"] ?? "default", models: models, options: options)
}

func retrievalMode(_ name: String) -> RetrievalMode? {
    switch name {
    case "keyword": .keyword
    case "vector": .vector
    case "hybrid": .hybrid
    default: nil
    }
}

/// "0.6,0.35" → (supported, weak).
func thresholds(_ arguments: Arguments) -> (supported: Float, weak: Float)? {
    guard let text = arguments["--thresholds"] else { return nil }
    let parts = text.split(separator: ",").compactMap { Float($0) }
    guard parts.count == 2, parts[0] >= parts[1] else { fail("--thresholds expects “supported,weak” with supported ≥ weak, for example 0.6,0.35.") }
    return (parts[0], parts[1])
}

func printStats(_ kb: KnowledgeBase) async {
    let stats = await kb.stats
    print("Knowledge base: \(stats.documentCount) documents, \(stats.chunkCount) chunks, \(stats.vectorCount) vectors.")
}

func runEval(_ arguments: Arguments) async throws {
    guard let dataset = arguments["--dataset"] else { fail(usage) }
    var modes: [EvalMode]?  // nil: keyword, vector, hybrid, and hybrid+rerank when a reranker is configured
    if let list = arguments["--modes"], list != "all" {
        let parsed = list.split(separator: ",").compactMap { EvalMode(name: String($0)) }
        guard parsed.count == list.split(separator: ",").count, !parsed.isEmpty else { fail("Unknown mode in --modes “\(list)”.") }
        modes = parsed
    }

    var ingest: IngestOptions?
    if arguments["--chunk-tokens"] != nil || arguments["--overlap-tokens"] != nil {
        let defaults = IngestOptions.default
        ingest = IngestOptions(
            targetChunkTokens: arguments.integer("--chunk-tokens") ?? defaults.targetChunkTokens,
            overlapTokens: arguments.integer("--overlap-tokens") ?? defaults.overlapTokens)
    }

    // A scratch knowledge base, on the CPU unless --gpu, so runs are repeatable and leave nothing behind.
    let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("loci-eval-\(UUID().uuidString)", isDirectory: true)
    let kb = try await KnowledgeBase(
        directory: scratch, models: modelSet(arguments), options: knowledgeBaseOptions(arguments, gpu: arguments["--gpu"] == nil ? .none : .all))
    let report: EvalReport
    do {
        var sections: EvalSections = arguments["--no-retrieval"] == nil ? [.retrieval] : []
        if arguments["--answers"] != nil { sections.insert(.answers) }
        if arguments["--verifier"] != nil || arguments["--calibrate"] != nil { sections.insert(.verifier) }
        if sections.isEmpty { fail("Nothing to evaluate: --no-retrieval needs --answers or --verifier.") }
        var verifier = VerifierOptions(calibrate: arguments["--calibrate"] != nil, minimumPrecision: arguments["--min-precision"].flatMap(Double.init) ?? 0.95)
        var answerOptions: AnswerOptions?
        if let cutoffs = thresholds(arguments) {
            verifier.cutoffs = SupportCutoffs(supported: cutoffs.supported, weak: cutoffs.weak)
            var options = AnswerOptions(retrieval: RetrievalOptions(limit: arguments.integer("--limit") ?? 5), maxAnswerTokens: arguments.integer("--max-tokens") ?? 300, temperature: 0)
            options.supportThresholds = cutoffs
            answerOptions = options
        }
        report = try await Evaluator(knowledgeBase: kb, modes: modes, ingestOptions: ingest)
            .run(dataset: .folder(URL(fileURLWithPath: dataset, isDirectory: true)), sections: sections, answerOptions: answerOptions, verifier: verifier)
    } catch {
        try? await kb.destroy()
        throw error
    }
    try await kb.destroy()

    print(report.markdown())
    if let out = arguments["--out"] { try report.write(to: URL(fileURLWithPath: out), format: .markdown) }
    if let json = arguments["--json"] { try report.write(to: URL(fileURLWithPath: json), format: .json) }

    if let baselinePath = arguments["--fail-on-regression"] {
        let baseline = try EvalReport.read(from: URL(fileURLWithPath: baselinePath))
        let margin = (arguments["--margin"].flatMap(Double.init) ?? 2) / 100
        let regressions = report.regressions(against: baseline, margin: margin)
        if !regressions.isEmpty {
            fail("Regression against \(baselinePath):\n" + regressions.map { "  - \($0)" }.joined(separator: "\n"), code: 2)
        }
        print("No Recall@5 regression against \(baselinePath).")
    }
}

func runBench(_ arguments: Arguments) async throws {
    let models = modelSet(arguments)
    let options = knowledgeBaseOptions(arguments, gpu: arguments["--cpu"] == nil ? .all : .none)
    let benchmark = Benchmark {
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("loci-bench-\(UUID().uuidString)", isDirectory: true)
        return try await KnowledgeBase(directory: scratch, models: models, options: options)
    }
    let report = try await benchmark.run(BenchmarkOptions(
        chunkCount: arguments.integer("--chunks") ?? 10_000, queries: arguments.integer("--queries") ?? 100,
        rerankQueries: arguments.integer("--rerank-queries") ?? 10,
        answerDataset: arguments["--dataset"].map { URL(fileURLWithPath: $0, isDirectory: true) }))
    let markdown = Benchmark.markdown(report)
    print(markdown)
    if let out = arguments["--out"] { try Data(markdown.utf8).write(to: URL(fileURLWithPath: out), options: .atomic) }
    if let json = arguments["--json"] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(report).write(to: URL(fileURLWithPath: json), options: .atomic)
    }
}

func runModels(_ arguments: Arguments) async throws {
    let downloader = ModelDownloader()
    switch arguments.positional.first {
    case "list":
        let files = (try? FileManager.default.contentsOfDirectory(at: downloader.directory, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        let models = files.filter { $0.pathExtension == "gguf" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        if models.isEmpty { print("No models in \(downloader.directory.path)") }
        for url in models {
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            print("\(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file))  \(url.lastPathComponent)")
        }
    case "download":
        guard arguments.positional.count == 3 else { fail(usage) }
        let source = ModelSource.hub(repo: arguments.positional[1], file: arguments.positional[2], sha256: arguments["--sha256"])
        var lastPercent = -1
        for try await event in downloader.download(source) {
            switch event {
            case .progress(let progress):
                if let fraction = progress.fraction, Int(fraction * 100) != lastPercent {
                    lastPercent = Int(fraction * 100)
                    let received = ByteCountFormatter.string(fromByteCount: progress.receivedBytes, countStyle: .file)
                    FileHandle.standardError.write(Data("\r\(lastPercent)%  \(received)   ".utf8))
                }
            case .finished(let url):
                FileHandle.standardError.write(Data("\n".utf8))
                print("Saved \(url.path)" + (arguments["--sha256"] == nil ? " (not verified: no --sha256 given)" : " (SHA-256 verified)"))
            }
        }
    default:
        fail(usage)
    }
}

func run() async throws {
    let raw = Array(CommandLine.arguments.dropFirst())
    guard let command = raw.first, let arguments = Arguments(Array(raw.dropFirst())) else { fail(usage) }

    if command == "bench" {
        try await runBench(arguments)
        return
    }
    if command == "models" {
        try await runModels(arguments)
        return
    }
    if command == "eval" {
        try await runEval(arguments)
        return
    }
    let kb = try await openKnowledgeBase(arguments)

    switch command {
    case "index":
        guard let path = arguments.positional.first else { fail(usage) }
        let url = URL(fileURLWithPath: path)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            fail("No such file or folder: \(path)")
        }
        if isDirectory.boolValue {
            var added = 0, skipped = 0
            for try await progress in kb.add(contentsOf: url) {
                switch progress.phase {
                case .done:
                    added += 1
                    print("[\(progress.completed)/\(progress.total)] added   \(progress.document)")
                case .skipped(let reason):
                    skipped += 1
                    print("[\(progress.completed)/\(progress.total)] skipped \(progress.document) — \(reason)")
                default:
                    break
                }
            }
            print("Done: \(added) added, \(skipped) skipped.")
        } else {
            let document = try await kb.add(.file(url))
            print("Added “\(document.title)” (\(document.chunkCount) chunks).")
        }
        await printStats(kb)

    case "search":
        let query = arguments.positional.joined(separator: " ")
        guard !query.isEmpty else { fail(usage) }
        guard let mode = retrievalMode(arguments["--mode"] ?? "hybrid") else { fail("Unknown --mode. Use keyword, vector or hybrid.") }
        let limit = arguments.integer("--limit") ?? 5
        let results = try await kb.search(
            query, options: RetrievalOptions(limit: limit, mode: mode, rerank: arguments["--no-rerank"] == nil))
        if results.isEmpty { print("No results.") }
        for (index, result) in results.enumerated() {
            let page = result.page.map { " · page \($0)" } ?? ""
            var scores: [String] = []
            if let value = result.scores.rerank { scores.append(String(format: "rerank %.2f", value)) }
            if let value = result.scores.fused { scores.append(String(format: "fused %.4f", value)) }
            if let value = result.scores.bm25 { scores.append(String(format: "bm25 %.2f", value)) }
            if let value = result.scores.vector { scores.append(String(format: "cosine %.3f", value)) }
            print("[\(index + 1)] \(result.document.title)\(page) · \(scores.joined(separator: ", "))")
            let flat = result.text.replacingOccurrences(of: "\n\n", with: " ¶ ")
            let excerpt = flat.count > 280 ? flat.prefix(280) + "…" : flat
            print("    \(excerpt)\n")
        }

    case "ask":
        let question = arguments.positional.joined(separator: " ")
        guard !question.isEmpty else { fail(usage) }
        var options = AnswerOptions(
            retrieval: RetrievalOptions(limit: arguments.integer("--limit") ?? 5, rerank: arguments["--no-rerank"] == nil),
            maxAnswerTokens: arguments.integer("--max-tokens") ?? 400,
            temperature: arguments["--temperature"].flatMap(Float.init) ?? 0.2)
        options.instructions = arguments["--instructions"]
        options.verify = arguments["--no-verify"] == nil
        options.refuseWhenUnsupported = arguments["--keep-unsupported"] == nil
        if let cutoffs = thresholds(arguments) { options.supportThresholds = cutoffs }
        let streaming = arguments["--stream"] != nil
        var answer: Answer?
        for try await event in kb.streamAnswer(question, options: options) {
            switch event {
            case .retrieved(let results):
                if streaming { print("Retrieved \(results.count) passages…\n") }
            case .token(let piece):
                if streaming { print(piece, terminator: ""); fflush(stdout) }
            case .sentence(let sentence):
                if streaming { print("\n  [\(sentence.support.rawValue) \(String(format: "%.2f", sentence.score))] \(sentence.text)") }
            case .finished(let finished):
                answer = finished
            }
        }
        guard let answer else { fail("No answer was produced.") }
        if streaming { print("\n\n--- final answer (citations cleaned and renumbered) ---") }
        print(answer.text)
        if answer.refused {
            print("\n(The documents do not contain an answer.)")
        } else if !answer.citations.isEmpty {
            print("\nSources:")
            for citation in answer.citations {
                let page = citation.page.map { " · page \($0)" } ?? ""
                let flat = citation.excerpt.replacingOccurrences(of: "\n\n", with: " ¶ ")
                print("[\(citation.index)] \(citation.document.title)\(page)")
                print("    “\(flat.count > 200 ? flat.prefix(200) + "…" : flat)”")
            }
        } else {
            print("\n(No sources were cited.)")
        }
        if let support = answer.support {
            print(String(format: "\nVerification: %.0f%% of %d sentences supported (%@)", support.supportedFraction * 100, answer.sentences.count, support.level.rawValue))
            for sentence in answer.sentences {
                let mark = switch sentence.support { case .supported: "✓"; case .weak: "~"; case .unsupported: "✗" }
                print("  \(mark) \(String(format: "%.2f", sentence.score))  \(sentence.text)")
            }
        }
        let stats = answer.stats
        func ms(_ d: Duration) -> String { String(format: "%.0f ms", Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15) }
        print("\nretrieval \(ms(stats.retrievalTime)) · rerank \(ms(stats.rerankTime)) · first token \(ms(stats.timeToFirstToken)) · "
            + String(format: "%.1f tok/s", stats.generationTokensPerSecond) + " · verify \(ms(stats.verificationTime)) · prompt \(stats.contextTokens) tokens" + (answer.support == nil && !answer.refused ? " · unverified" : ""))

    case "documents":
        for document in try await kb.documents() {
            let pages = document.pageCount.map { ", \($0) pages" } ?? ""
            print("\(document.title) — \(document.kind.rawValue), \(document.chunkCount) chunks\(pages)")
        }
        await printStats(kb)

    case "reindex":
        for try await progress in kb.reindex() {
            if case .done = progress.phase { print("[\(progress.completed)/\(progress.total)] \(progress.document)") }
        }
        await printStats(kb)

    default:
        fail(usage)
    }
}

do {
    try await run()
} catch {
    fail("error: \(error.localizedDescription)")
}
