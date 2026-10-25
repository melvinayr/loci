import Loci
import LociHub
import SwiftUI

/// Downloads the models on first launch, opens the knowledge base and shows the reader.
@MainActor
public final class ReaderBootstrap: ObservableObject {
    public enum State {
        case starting
        case downloading(model: String, fraction: Double?)
        case ready(ReaderModel)
        case failed(String)
    }

    @Published public private(set) var state: State = .starting
    private let catalogSet: CatalogSet

    public init(catalogSet: CatalogSet = ModelCatalog.recommended) {
        self.catalogSet = catalogSet
    }

    public func start() async {
        do {
            let downloader = ModelDownloader()
            try await downloader.ensure(catalogSet) { [weak self] model, progress in
                let fraction = progress.totalBytes.flatMap { $0 > 0 ? Double(progress.receivedBytes) / Double($0) : nil }
                Task { @MainActor in self?.state = .downloading(model: model.name, fraction: fraction) }
            }
            let support = try Self.supportDirectory()
            let knowledgeBase = try await KnowledgeBase(directory: support.appendingPathComponent("KnowledgeBase"), models: catalogSet.set)
            let engine = KnowledgeBaseEngine(knowledgeBase: knowledgeBase, library: support.appendingPathComponent("Library"))
            state = .ready(ReaderModel(engine: engine))
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private static func supportDirectory() throws -> URL {
        let base = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let directory = base.appendingPathComponent("LociReader", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}

/// The view an app shows: setup progress, then the reader.
public struct ReaderRootView: View {
    @StateObject private var bootstrap = ReaderBootstrap()

    public init() {}

    public var body: some View {
        switch bootstrap.state {
        case .starting:
            ProgressView("Starting").task { await bootstrap.start() }
        case .downloading(let model, let fraction):
            VStack(spacing: 12) {
                Text("Downloading \(model)")
                if let fraction { ProgressView(value: fraction) } else { ProgressView() }
                Text("Models are downloaded once and stay on this device.").font(.caption).foregroundColor(.secondary)
            }
            .padding()
            .frame(maxWidth: 360)
        case .ready(let model):
            ReaderView(model: model)
        case .failed(let message):
            VStack(spacing: 12) {
                Text("Setup failed").font(.headline)
                Text(message).foregroundColor(.secondary)
                Button("Try again") { Task { await bootstrap.start() } }
            }
            .padding()
        }
    }
}
