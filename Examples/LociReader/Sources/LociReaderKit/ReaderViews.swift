import Loci
import PDFKit
import SwiftUI

/// The reader's main screen: library on the left, question and answer in the middle, the cited source on the right.
public struct ReaderView: View {
    @ObservedObject var model: ReaderModel
    @State private var importing = false

    public init(model: ReaderModel) {
        self.model = model
    }

    public var body: some View {
        NavigationView {
            LibraryView(model: model, importing: $importing)
            AskView(model: model)
            SourceView(content: model.source)
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.pdf, .plainText, .html, .text, .data], allowsMultipleSelection: true) { result in
            if case .success(let urls) = result { Task { await model.importFiles(urls) } }
        }
        .task { await model.refresh() }
    }
}

struct LibraryView: View {
    @ObservedObject var model: ReaderModel
    @Binding var importing: Bool

    var body: some View {
        List {
            Section("Documents") {
                ForEach(model.documents, id: \.id) { document in
                    Label(document.title, systemImage: document.kind == .pdf ? "doc.richtext" : "doc.text")
                        .contextMenu { Button("Remove", role: .destructive) { Task { await model.remove(document) } } }
                }
                if model.documents.isEmpty { Text("Add a PDF, Markdown, HTML or text file to begin.").foregroundColor(.secondary) }
            }
            switch model.importState {
            case .idle: EmptyView()
            case .importing(let current, let completed, let total):
                Section { ProgressView("Adding \(current) (\(completed + 1) of \(total))") }
            case .finished(let added, let problems):
                Section("Last import") {
                    Text("\(added) added")
                    ForEach(problems, id: \.self) { Text($0).foregroundColor(.red).font(.caption) }
                }
            }
        }
        .navigationTitle("Library")
        .toolbar { Button { importing = true } label: { Label("Add files", systemImage: "plus") } }
    }
}

struct AskView: View {
    @ObservedObject var model: ReaderModel

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    AnswerCard(model: model)
                    if !model.sources.isEmpty { SourceList(model: model) }
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider()
            HStack {
                TextField("Ask your documents", text: $model.question).onSubmit { model.ask() }.textFieldStyle(.roundedBorder)
                if model.phase.isBusy {
                    Button("Stop") { model.cancel() }
                } else {
                    Button("Ask") { model.ask() }.disabled(!model.canAsk)
                }
            }
            .padding()
        }
        .navigationTitle("Ask")
    }
}

struct AnswerCard: View {
    @ObservedObject var model: ReaderModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            switch model.phase {
            case .idle: EmptyView()
            case .searching: ProgressView("Searching")
            case .failed(let message): Text(message).foregroundColor(.red)
            case .writing, .checking, .finished:
                if model.sentences.isEmpty {
                    Text(model.draft)
                } else {
                    ForEach(Array(model.sentences.enumerated()), id: \.offset) { _, sentence in
                        HStack(alignment: .firstTextBaseline, spacing: 6) {
                            SupportBadge(level: sentence.support)
                            Text(AnswerRendering.attributed(sentence.text))
                        }
                    }
                }
                if model.phase == .checking { ProgressView("Checking sentences against their sources") }
                if let summary = model.supportSummary { Text(summary).font(.caption).foregroundColor(.secondary) }
            }
        }
        .environment(\.openURL, OpenURLAction { url in
            guard let index = AnswerRendering.citationIndex(from: url) else { return .systemAction }
            Task { await model.showCitation(index) }
            return .handled
        })
        .textSelection(.enabled)
    }
}

struct SupportBadge: View {
    let level: SupportLevel

    var body: some View {
        switch level {
        case .supported: Image(systemName: "checkmark.circle.fill").foregroundColor(.green).accessibilityLabel("Supported")
        case .weak: Image(systemName: "questionmark.circle.fill").foregroundColor(.orange).accessibilityLabel("Weakly supported")
        case .unsupported: Image(systemName: "xmark.circle.fill").foregroundColor(.red).accessibilityLabel("Unsupported")
        }
    }
}

struct SourceList: View {
    @ObservedObject var model: ReaderModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Looked at").font(.headline)
            ForEach(Array(model.sources.enumerated()), id: \.offset) { _, result in
                Button { Task { await model.showSource(result) } } label: {
                    VStack(alignment: .leading) {
                        Text(result.document.title + (result.page.map { " · page \($0)" } ?? "")).font(.subheadline)
                        Text(result.text).font(.caption).foregroundColor(.secondary).lineLimit(2)
                    }
                }
                .buttonStyle(.plain)
            }
        }
    }
}

struct SourceView: View {
    let content: SourceContent?

    var body: some View {
        Group {
            switch content {
            case nil:
                Text("Tap a citation to see its source.").foregroundColor(.secondary)
            case .pdf(let url, let page, let excerpt):
                PDFSourceView(url: url, page: page, excerpt: excerpt)
            case .text(let title, let highlighted):
                ScrollView { Text(highlighted.text).padding().frame(maxWidth: .infinity, alignment: .leading) }.navigationTitle(title)
            case .excerptOnly(let title, let page, let excerpt):
                VStack(alignment: .leading, spacing: 8) {
                    Text(title + (page.map { " · page \($0)" } ?? "")).font(.headline)
                    Text("The original file is no longer available; this is the cited passage.").font(.caption).foregroundColor(.secondary)
                    Text(excerpt)
                }.padding()
            }
        }
        .navigationTitle("Source")
    }
}

#if canImport(AppKit)
import AppKit
typealias PlatformViewRepresentable = NSViewRepresentable
#else
import UIKit
typealias PlatformViewRepresentable = UIViewRepresentable
#endif

/// A PDFKit view opened at the cited page with the passage marked.
struct PDFSourceView: PlatformViewRepresentable {
    let url: URL
    let page: Int
    let excerpt: String

    private func configure(_ view: PDFView) {
        if view.document?.documentURL != url { view.document = PDFDocument(url: url) }
        view.autoScales = true
        guard let document = view.document, let target = PDFHighlighter.target(in: document, page: page, excerpt: excerpt) else { return }
        view.go(to: target.page)
        if let selection = target.selection {
            view.setCurrentSelection(selection, animate: true)
            view.go(to: selection)
        }
    }

    #if canImport(AppKit)
    func makeNSView(context: Context) -> PDFView { let view = PDFView(); configure(view); return view }
    func updateNSView(_ view: PDFView, context: Context) { configure(view) }
    #else
    func makeUIView(context: Context) -> PDFView { let view = PDFView(); configure(view); return view }
    func updateUIView(_ view: PDFView, context: Context) { configure(view) }
    #endif
}
