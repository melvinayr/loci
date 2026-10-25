# Building a document reader

A walkthrough of `Examples/LociReader`, a SwiftUI app that imports files, streams answers and opens a citation at its page.

## 1. Set up

`ReaderBootstrap` downloads the models with `ModelDownloader`, opens a ``KnowledgeBase`` in Application Support and wraps it in a `KnowledgeBaseEngine`. Imported files are copied into a library folder so they stay readable later; on iOS that also leaves the picker's security scope behind.

## 2. Import

`ReaderModel.importFiles(_:)` adds each file and keeps going when one fails, reporting a line per failure.

## 3. Stream the answer

`ReaderModel.ask()` consumes ``KnowledgeBase/streamAnswer(_:options:)``: sources appear on `.retrieved`, the reply grows on `.token`, sentences get a ✓, ? or ✗ badge on `.sentence`, and `.finished` carries the final text with renumbered citations. `[n]` markers become links, and tapping one calls `showCitation(_:)`.

## 4. Show the source

For a PDF, the view opens the file at ``Citation/page`` and finds the passage on that page. PDFKit's text differs from Loci's normalised text (line breaks, ligatures), so `PassageLocator` matches on letters and digits only and maps back to the page's own offsets; if the passage cannot be found the page is still shown. For other formats `HighlightedText` marks the exact ``Citation/range`` in the normalised text.

## Running it

```bash
cd Examples/LociReader
swift run LociReader
```

On iOS, add `LociReaderKit` to an app target in Xcode and show `ReaderRootView`.
