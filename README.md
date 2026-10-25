<h1 align="center">Loci</h1>

<p align="center">
  <b>Grounded, cited answers from your documents — entirely on device.</b><br>
  On-device RAG for Swift with verified citations, powered by llama.cpp and Metal.
</p>

<p align="center">
  <a href="https://swiftpackageindex.com/melvinayr/loci"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmelvinayr%2Floci%2Fbadge%3Ftype%3Dswift-versions" alt="Swift versions"></a>
  <a href="https://swiftpackageindex.com/melvinayr/loci"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmelvinayr%2Floci%2Fbadge%3Ftype%3Dplatforms" alt="Platforms"></a>
  <a href="https://github.com/melvinayr/loci/actions"><img src="https://github.com/melvinayr/loci/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
</p>

```swift
import Loci

let kb = try await KnowledgeBase(directory: storeURL, models: .recommended)
try await kb.add(.file(trialPDF))

let answer = try await kb.ask("What sample size did the trial use?")
// "The trial enrolled 412 participants [1]."
// [1] trial.pdf · page 4 · "…412 participants were randomised…"  ✓ supported
```


---

## Why Loci?

The *method of loci* is the memory-palace technique: you store knowledge in places and recall it by walking back to them. Loci stores your documents on device and, for every answer, takes you back to the exact place it came from.

- **Every claim is cited** — to the document, page and passage.
- **Every citation is checked** — a cross-encoder scores whether the passage actually supports the sentence. Unsupported claims are flagged, and if the documents don't contain the answer, Loci says so instead of guessing.
- **The whole pipeline, one package** — parsing, chunking, hybrid search (BM25 + vectors), reranking, generation and verification.
- **Quality you can measure** — the built-in `LociEval` harness reports retrieval recall, citation accuracy, faithfulness, latency and memory on your own documents.
- **Private by design** — no network calls in the core library, iOS Data Protection on every file, no document text in logs.
- **Fits on a phone** — embedder, reranker and generator share one memory budget, running natively via llama.cpp and Metal.
- **Wide reach** — iOS 16.4+, macOS 13.3+. MIT licensed.

## How it works

```
documents → parse → chunk → embed → BM25 + vector search → rerank → generate with [n] → verify each sentence
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Requirements

| | Minimum |
|---|---|
| Swift | 6.0 |
| Xcode | 16 |
| iOS / iPadOS | 16.4 |
| macOS | 13.3 |

Metal acceleration needs a physical Apple Silicon device. The pinned llama.cpp binary has no iOS Simulator slice, so the package builds for macOS and physical iOS devices only.

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/melvinayr/loci.git", from: "0.1.2")
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "Loci", package: "loci"),
        // .product(name: "LociHub", package: "loci"),   // optional: download models
        // .product(name: "LociEval", package: "loci"),  // optional: evaluation
    ])
]
```

## Models

Loci uses three GGUF models. `ModelSet.recommended` (bge-small, Qwen3-Reranker-0.6B, Qwen3-4B; about 3.2 GB, MIT and Apache-2.0) suits iPhones with 6 GB or more and Macs; `ModelSet.quality` swaps in a larger embedder. Both are English-first. See [docs/MODELS.md](docs/MODELS.md) for the measurements behind the choice, sizes, licences and calibrated thresholds.

| Role | What it does |
|---|---|
| Embedder | Turns passages and questions into vectors for semantic search |
| Reranker | Re-orders search results, then verifies citations |
| Generator | Writes the answer from the retrieved passages |

Bundle the files with your app, or download them at runtime with `LociHub` (`ModelDownloader().ensure(ModelCatalog.recommended)`: resumable, checksum-verified, foreground only). Check each model's licence before shipping.

## Usage

### Add documents

```swift
try await kb.add(.file(pdfURL))
try await kb.add(.text(noteBody, title: "Meeting notes"))

for try await progress in kb.add(contentsOf: folderURL) {
    print(progress.document, progress.phase)
}
```

PDF, Markdown, HTML and plain text. Re-adding an unchanged document is skipped automatically.

### Ask

```swift
let answer = try await kb.ask("What were the side effects reported?")

print(answer.text)
for citation in answer.citations {
    print("[\(citation.index)] \(citation.document.title) p.\(citation.page ?? 0): \(citation.excerpt)")
}
if let support = answer.support { print("Supported: \(Int(support.supportedFraction * 100))%") }
```

### Stream (great for SwiftUI)

```swift
for try await event in kb.streamAnswer(question) {
    switch event {
    case .retrieved(let sources): self.sources = sources      // show sources immediately
    case .token(let text):        self.answer += text
    case .sentence(let s):        self.badges.append(s.support) // ✓ supported / ⚠ weak / ✗ unsupported
    case .finished(let a):        self.final = a
    }
}
```

### Search only

```swift
let results = try await kb.search("dosage guidelines", options: .init(limit: 10))
```

### Evaluate

```bash
swift run loci-cli eval --dataset ./my-eval --models recommended --out report.md
```

See [docs/EVALUATION.md](docs/EVALUATION.md).

## Privacy

- The `Loci` and `LociEval` products contain no networking code (enforced in CI).
- Databases are protected with iOS Data Protection (`completeUntilFirstUserAuthentication` by default, configurable).
- Logs contain IDs, counts and timings — never document text, questions or answers.
- `kb.destroy()` removes everything.

## Memory on iOS

Loci checks available memory before loading a model and throws `LociError.insufficientMemory` instead of letting iOS terminate your app. The default `.lowMemory` residency loads one model at a time. For larger models, add the **Increased Memory Limit** entitlement. See the Memory guide in the docs.

## Documentation

Full API reference, articles and a walkthrough of the reader app are in the DocC catalogue (`Sources/Loci/Loci.docc`), hosted by Swift Package Index once the package is published. Note that SPI builds for macOS and iOS devices, and this package has no iOS Simulator slice.

## Examples

- [`Examples/LociReader`](Examples/LociReader) — SwiftUI document reader: ask questions, tap a citation to jump to the highlighted passage.
- [`Examples/loci-cli`](Examples/loci-cli) — index, search, ask, eval and bench from the terminal.

## FAQ

**Does it work offline?** Yes. Once models are on the device, nothing touches the network.

**How accurate is it?** It depends on your documents and models, which is why Loci ships an evaluation harness. Measured results for the default models are in [docs/MODELS.md](docs/MODELS.md) and speed on one Mac in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). The support badge is a relevance check: it catches claims unrelated to their source but not on-topic contradictions, so use the citation to verify ([how verification works](Sources/Loci/Loci.docc/HowVerificationWorks.md)).

**Can I use Apple's Foundation Models for the answer?** Planned for v0.2 via a `Generator` protocol.

**Scanned PDFs?** Not yet — Loci reports them as unreadable. OCR is on the roadmap.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.

## License

MIT — see [LICENSE](LICENSE).
