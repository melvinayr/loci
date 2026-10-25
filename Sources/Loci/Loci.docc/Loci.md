# ``Loci``

Answer questions from your own documents, on the device, with citations you can check.

## Overview

Loci indexes PDFs, Markdown, HTML and text files, retrieves the passages that matter, and has a local model write an answer in which every claim points at the exact passage it came from. Each sentence is then checked against its sources, and the answer says how well it is supported or declines to answer. Nothing leaves the device.

```swift
let kb = try await KnowledgeBase(directory: storeURL, models: .recommended)
try await kb.add(.file(reportURL))

let answer = try await kb.ask("How many participants completed the trial?")
print(answer.text)
for citation in answer.citations {
    print(citation.index, citation.document.title, citation.page ?? 0, citation.excerpt)
}
```

## Topics

### Essentials

- <doc:GettingStarted>
- ``KnowledgeBase``
- ``ModelSet``

### Answers and citations

- ``Answer``
- ``Citation``
- ``AnswerEvent``
- ``AnswerOptions``
- <doc:HowVerificationWorks>

### Choosing and running models

- <doc:ChoosingModels>
- <doc:MemoryOnIOS>
- ``ModelResidency``

### Measuring quality

- <doc:EvaluatingYourKnowledgeBase>

### Building an app

- <doc:BuildingADocumentReader>
