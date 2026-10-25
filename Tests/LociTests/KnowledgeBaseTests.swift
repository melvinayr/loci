import Foundation
import Testing
@testable import Loci
@testable import LociLlama
@testable import LociStore

private let keyword = RetrievalOptions(mode: .keyword)

/// Creates a knowledge base in a temporary directory and removes it afterwards.
private func withKnowledgeBase(embedder: FakeEmbedder = FakeEmbedder(), _ body: (KnowledgeBase, URL) async throws -> Void) async throws {
    let directory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let kb = try await KnowledgeBase(directory: directory.appendingPathComponent("kb"), services: await ModelServices(embedder: embedder))
    try await body(kb, directory)
}

/// The citation invariants, checked over everything stored:
/// every chunk range slices to exactly its text, and every PDF chunk's page contains its range.
private func assertStoredInvariants(_ kb: KnowledgeBase, sourceLocation: SourceLocation = #_sourceLocation) async throws {
    for document in try await kb.documents() {
        let text = try #require(try await kb.documentText(for: document.id), sourceLocation: sourceLocation)
        let chunks = try await kb.store.chunks(documentID: document.id.rawValue)
        #expect(chunks.count == document.chunkCount, sourceLocation: sourceLocation)
        let pageRanges = try await kb.store.pageRanges(documentID: document.id.rawValue)
        for chunk in chunks {
            #expect(chunk.text == text.scalarSlice(chunk.start..<chunk.end), "chunk \(chunk.ordinal) of \(document.title)", sourceLocation: sourceLocation)
            if document.kind == .pdf {
                let page = try #require(chunk.page, sourceLocation: sourceLocation)
                let range = try #require(pageRanges?[page - 1], sourceLocation: sourceLocation)
                #expect(range.lowerBound <= chunk.start && chunk.end <= range.upperBound, "page \(page) must contain chunk \(chunk.ordinal)", sourceLocation: sourceLocation)
            } else {
                #expect(chunk.page == nil, sourceLocation: sourceLocation)
            }
        }
    }
}

private func assertResultInvariants(_ results: [SearchResult], _ kb: KnowledgeBase, sourceLocation: SourceLocation = #_sourceLocation) async throws {
    for result in results {
        let text = try #require(try await kb.documentText(for: result.document.id), sourceLocation: sourceLocation)
        #expect(text.scalarSlice(result.range) == result.text, sourceLocation: sourceLocation)
        if result.document.kind == .pdf {
            let page = try #require(result.page, sourceLocation: sourceLocation)
            let ranges = try #require(try await kb.store.pageRanges(documentID: result.document.id.rawValue), sourceLocation: sourceLocation)
            let pageRange = try #require(ranges[page - 1], sourceLocation: sourceLocation)
            #expect(pageRange.lowerBound <= result.range.lowerBound && result.range.upperBound <= pageRange.upperBound, sourceLocation: sourceLocation)
        }
    }
}

@Suite("KnowledgeBase ingest")
struct KnowledgeBaseIngestTests {
    @Test func addsEveryFormatAndKeepsInvariants() async throws {
        try await withKnowledgeBase { kb, directory in
            let pdf = directory.appendingPathComponent("report.pdf")
            try writePDF(pages: [
                "Page one introduces the randomised trial.",
                "Page two reports that a quokka was observed.\n\nA second paragraph on page two.",
            ], to: pdf)

            for name in ["sample.md", "sample.html", "sample.txt", "long.md"] {
                try await kb.add(.file(Fixtures.url(name)))
            }
            let report = try await kb.add(.file(pdf))
            try await kb.add(.text("Inline note about participants.", title: "Note", uri: "note://1"))

            let documents = try await kb.documents()
            #expect(documents.count == 6)
            #expect(Set(documents.map(\.kind)) == [.markdown, .html, .text, .pdf])
            #expect(report.pageCount == 2 && report.kind == .pdf && report.title == "report")
            #expect(documents.contains { $0.title == "Field Notes" } && documents.contains { $0.title == "Trial & Results" })

            try await assertStoredInvariants(kb)

            let results = try await kb.search("quokka", options: RetrievalOptions(limit: 10, mode: .keyword))
            #expect(results.first?.document.id == report.id)
            #expect(results.first?.page == 2)
            try await assertResultInvariants(results, kb)
        }
    }

    @Test func reAddingAnUnchangedDocumentIsANoOp() async throws {
        try await withKnowledgeBase { kb, _ in
            let first = try await kb.add(.file(Fixtures.url("sample.md")))
            let before = await kb.stats
            let again = try await kb.add(.file(Fixtures.url("sample.md")))
            let afterOptions = try await kb.add(.file(Fixtures.url("sample.md")), options: IngestOptions(targetChunkTokens: 10))
            let after = await kb.stats

            #expect(again == first)
            #expect(afterOptions.id == first.id)
            #expect(before.documentCount == 1 && after.documentCount == 1)
            #expect(before.chunkCount == after.chunkCount)

            let rows = try await kb.store.chunks(documentID: first.id.rawValue).count
            #expect(rows == first.chunkCount)
        }
    }

    @Test func aChangedFileReplacesItsPreviousVersion() async throws {
        try await withKnowledgeBase { kb, directory in
            let file = directory.appendingPathComponent("notes.txt")
            try "The old content mentions zebras.".write(to: file, atomically: true, encoding: .utf8)
            let old = try await kb.add(.file(file))
            try "The new content mentions giraffes.".write(to: file, atomically: true, encoding: .utf8)
            let new = try await kb.add(.file(file))

            #expect(old.id != new.id)
            #expect(try await kb.documents().map(\.id) == [new.id])
            #expect(try await kb.search("zebras", options: keyword).isEmpty)
            #expect(try await kb.search("giraffes", options: keyword).count == 1)
        }
    }

    @Test func removeDeletesDocumentAndItsPassages() async throws {
        try await withKnowledgeBase { kb, _ in
            let doc = try await kb.add(.text("Unique zanzibar content.", title: "T"))
            #expect(try await kb.search("zanzibar", options: keyword).count == 1)
            try await kb.remove(doc.id)
            #expect(try await kb.documents().isEmpty)
            #expect(try await kb.search("zanzibar", options: keyword).isEmpty)
            #expect(await kb.stats.chunkCount == 0)
            try await kb.remove(doc.id)  // removing twice is harmless
        }
    }

    @Test func folderIngestReportsProgressAndSkipsBadFilesWithoutStopping() async throws {
        try await withKnowledgeBase { kb, directory in
            let folder = directory.appendingPathComponent("docs")
            try FileManager.default.createDirectory(at: folder.appendingPathComponent("sub"), withIntermediateDirectories: true)
            try "alpha document".write(to: folder.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
            try "# Beta\n\nbeta document".write(to: folder.appendingPathComponent("sub/b.md"), atomically: true, encoding: .utf8)
            try Data([0, 1, 2]).write(to: folder.appendingPathComponent("c.docx"))
            try "   ".write(to: folder.appendingPathComponent("empty.txt"), atomically: true, encoding: .utf8)
            try "hidden".write(to: folder.appendingPathComponent(".hidden.txt"), atomically: true, encoding: .utf8)
            try "alpha document".write(to: folder.appendingPathComponent("dup.txt"), atomically: true, encoding: .utf8)

            var events: [IngestProgress] = []
            for try await progress in kb.add(contentsOf: folder) { events.append(progress) }

            let terminal = events.filter {
                switch $0.phase {
                case .done, .skipped: true
                default: false
                }
            }
            #expect(events.allSatisfy { $0.total == 5 }, "hidden files are not counted")
            #expect(terminal.count == 5)
            #expect(terminal.map(\.completed) == [1, 2, 3, 4, 5])

            func phase(_ name: String) -> IngestProgress.Phase? { terminal.first { $0.document == name }?.phase }
            #expect(phase("a.txt") == .done)
            #expect(phase("b.md") == .done)
            #expect(phase("dup.txt") == .skipped("unchanged"))
            if case .skipped(let reason)? = phase("c.docx") { #expect(reason.contains("not supported")) } else { Issue.record("c.docx not skipped") }
            if case .skipped? = phase("empty.txt") {} else { Issue.record("empty.txt not skipped") }

            #expect(try await kb.documents().count == 2)
            try await assertStoredInvariants(kb)
        }
    }

    @Test func cancellingAFolderIngestLeavesTheStoreConsistent() async throws {
        try await withKnowledgeBase { kb, directory in
            let folder = directory.appendingPathComponent("many")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for i in 0..<60 {
                let body = (0..<200).map { "Document \(i) sentence \($0) about topic \($0 % 9)." }.joined(separator: " ")
                try body.write(to: folder.appendingPathComponent("d\(i).txt"), atomically: true, encoding: .utf8)
            }
            let task = Task {
                for try await _ in kb.add(contentsOf: folder) {}
            }
            try await Task.sleep(for: .milliseconds(30))
            task.cancel()
            _ = await task.result

            // The producer finishes the document it is on after the consumer cancels; wait for it to go quiet.
            var stats = await kb.stats
            for _ in 0..<50 {
                try await Task.sleep(for: .milliseconds(100))
                let latest = await kb.stats
                if latest == stats { break }
                stats = latest
            }
            try await assertStoredInvariants(kb)  // every stored document is complete
            // The full-text index agrees with the chunk table.
            let hits = try await kb.store.keywordSearch("topic", limit: 100_000)
            #expect(hits.count == stats.chunkCount)
        }
    }

    @Test func errorsMapToLociError() async throws {
        try await withKnowledgeBase { kb, directory in
            await #expect(throws: LociError.self) { try await kb.add(.file(directory.appendingPathComponent("missing.txt"))) }
            await #expect(throws: LociError.self) { try await kb.add(.file(directory.appendingPathComponent("x.docx"))) }
            await #expect(throws: LociError.self) { try await kb.add(.text("  ", title: "Blank")) }
            do {
                _ = try await kb.add(.file(directory.appendingPathComponent("x.docx")))
            } catch let LociError.unsupportedDocument(url, reason) {
                #expect(url.lastPathComponent == "x.docx")
                #expect(!reason.isEmpty)
            }
        }
    }

    @Test func encryptedAndScannedPDFsAreReportedAsUnreadable() async throws {
        try await withKnowledgeBase { kb, directory in
            let locked = directory.appendingPathComponent("locked.pdf")
            try writePDF(pages: ["secret"], to: locked, password: "pw")
            let scan = directory.appendingPathComponent("scan.pdf")
            try writePDF(pages: [nil], to: scan)
            for url in [locked, scan] {
                do {
                    _ = try await kb.add(.file(url))
                    Issue.record("expected failure for \(url.lastPathComponent)")
                } catch LociError.documentUnreadable(let failed, _) {
                    #expect(failed == url)
                }
            }
            #expect(try await kb.documents().isEmpty)
        }
    }
}

@Suite("KnowledgeBase search")
struct KnowledgeBaseSearchTests {
    @Test func metadataAndDocumentFilters() async throws {
        try await withKnowledgeBase { kb, _ in
            let note = try await kb.add(.text("shared keyword in a note", title: "Note"), options: IngestOptions(metadata: ["type": "note"]))
            let paper = try await kb.add(.text("shared keyword in a paper", title: "Paper"), options: IngestOptions(metadata: ["type": "paper"]))

            func titles(_ filter: MetadataFilter?) async throws -> [String] {
                try await kb.search("shared keyword", options: RetrievalOptions(mode: .keyword, filter: filter)).map(\.document.title).sorted()
            }
            #expect(try await titles(nil) == ["Note", "Paper"])
            #expect(try await titles(.metadata("type", "note")) == ["Note"])
            #expect(try await titles(.metadata("type", "none")) == [])
            #expect(try await titles(.document([paper.id])) == ["Paper"])
            #expect(try await titles(.document([note.id, paper.id])) == ["Note", "Paper"])
            #expect(try await titles(.document([])) == [])
        }
    }

    @Test func limitScoresAndEmptyQueries() async throws {
        try await withKnowledgeBase { kb, _ in
            for i in 0..<8 { try await kb.add(.text("common term number \(i)", title: "D\(i)")) }
            let results = try await kb.search("common", options: RetrievalOptions(limit: 3, mode: .keyword))
            #expect(results.count == 3)
            #expect(results.allSatisfy { ($0.scores.bm25 ?? 0) > 0 && $0.scores.vector == nil })
            #expect(try await kb.search("", options: .default).isEmpty)
            #expect(try await kb.search("common", options: RetrievalOptions(limit: 0)).isEmpty)
            #expect(try await kb.search("nonexistentword", options: keyword).isEmpty)
        }
    }

    @Test func unicodeContentRoundTripsWithExactOffsets() async throws {
        try await withKnowledgeBase { kb, _ in
            let text = "😀 emoji start. Cafe\u{301} au lait is naïve. 日本語のテキストもあります。 Final sentence here."
            try await kb.add(.text(text, title: "Unicode"), options: IngestOptions(chunker: .sentence, targetChunkTokens: 8, overlapTokens: 0))
            try await assertStoredInvariants(kb)
            let results = try await kb.search("café", options: RetrievalOptions(mode: .keyword))
            #expect(!results.isEmpty)
            try await assertResultInvariants(results, kb)
        }
    }

    @Test func allChunkersKeepInvariantsThroughTheKnowledgeBase() async throws {
        for kind in [ChunkerKind.structural, .sentence, .tokenWindow] {
            try await withKnowledgeBase { kb, _ in
                try await kb.add(.file(Fixtures.url("long.md")), options: IngestOptions(chunker: kind, targetChunkTokens: 50, overlapTokens: 10))
                try await assertStoredInvariants(kb)
            }
        }
    }
}

@Suite("KnowledgeBase lifecycle")
struct KnowledgeBaseLifecycleTests {
    @Test func dataSurvivesReopening() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let location = directory.appendingPathComponent("kb")
        let fake = FakeEmbedder()
        do {
            let kb = try await KnowledgeBase(directory: location, services: await ModelServices(embedder: fake))
            try await kb.add(.text("persisted words", title: "P"))
            await kb.store.close()
        }
        let reopened = try await KnowledgeBase(directory: location, services: await ModelServices(embedder: fake))
        #expect(try await reopened.documents().map(\.title) == ["P"])
        #expect(try await reopened.search("persisted", options: keyword).count == 1)
        #expect(await reopened.stats.vectorCount == 1)
    }

    @Test func destroyRemovesEveryFileAndDisablesTheInstance() async throws {
        try await withKnowledgeBase { kb, directory in
            try await kb.add(.text("something", title: "T"))
            let location = directory.appendingPathComponent("kb")
            #expect(FileManager.default.fileExists(atPath: location.appendingPathComponent("loci.sqlite").path))
            try await kb.destroy()
            #expect(!FileManager.default.fileExists(atPath: location.path))
            await #expect(throws: LociError.self) { try await kb.documents() }
            await #expect(throws: LociError.self) { try await kb.add(.text("more", title: "T2")) }
            try await kb.destroy()  // idempotent
        }
    }

    @Test func nameMapsToAContainedDirectory() throws {
        for name in ["papers", "../../etc", "a/b", "..", ".", "", "name with spaces", "ünï"] {
            let url = try KnowledgeBase.defaultDirectory(for: name)
            let parent = url.deletingLastPathComponent()
            #expect(parent.lastPathComponent == "Loci", "“\(name)” escaped: \(url.path)")
            #expect(!["", ".", ".."].contains(url.lastPathComponent))
        }
        #expect(try KnowledgeBase.defaultDirectory(for: "a/b") != KnowledgeBase.defaultDirectory(for: "a_b"))
    }

    @Test func statsReflectContents() async throws {
        try await withKnowledgeBase { kb, _ in
            #expect(await kb.stats.documentCount == 0)
            try await kb.add(.file(Fixtures.url("sample.md")))
            let stats = await kb.stats
            #expect(stats.documentCount == 1 && stats.chunkCount > 0 && stats.databaseBytes > 0)
        }
    }

    @Test func corruptStoreSurfacesAsStoreCorrupted() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let location = directory.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: location, withIntermediateDirectories: true)
        try Data(repeating: 0x42, count: 2048).write(to: location.appendingPathComponent("loci.sqlite"))
        do {
            _ = try await KnowledgeBase(directory: location, services: await ModelServices(embedder: FakeEmbedder()))
            Issue.record("expected failure")
        } catch LociError.storeCorrupted {
        }
    }
}
