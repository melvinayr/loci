# Getting started

Add the package, download the models once, index files, ask.

## Add the package

Add `https://github.com/melvinayr/loci` in Xcode or in `Package.swift`, and depend on the `Loci` product. Add `LociHub` if the app should download models, and `LociEval` to measure quality. Loci needs macOS 13.3 or iOS 16.4.

## Get the models

``ModelSet/recommended`` names the models by Hugging Face repository and checksum. `LociHub` downloads them into Application Support, resuming interrupted downloads and verifying the checksum:

```swift
import LociHub

let downloader = ModelDownloader()
try await downloader.ensure(ModelCatalog.recommended) { model, progress in
    print(model.name, progress.fraction ?? 0)
}
```

The set is about 3.2 GB; show the size before downloading on a phone. Without `LociHub`, point a ``ModelSet`` at files (`.file(url)`) or bundled resources.

## Index and ask

```swift
let kb = try await KnowledgeBase(directory: storeURL, models: .recommended)
for try await progress in kb.add(contentsOf: documentsFolder) { print(progress) }

for try await event in kb.streamAnswer("What was the dropout rate?") {
    switch event {
    case .retrieved(let sources): show(sources)       // before generation starts
    case .token(let piece): append(piece)             // the reply as it is written
    case .sentence(let sentence): mark(sentence)      // each sentence once checked
    case .finished(let answer): present(answer)       // final text, renumbered citations
    }
}
```

Adding a file that has not changed is a no-op. Citations carry the document, the page (for PDFs), the character range in the normalised text and the excerpt, and `range` always slices to exactly `excerpt`.
