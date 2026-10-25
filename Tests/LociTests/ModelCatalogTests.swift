import Foundation
import LociHub
import Loci
import Testing

@Suite struct ModelCatalogTests {
    @Test func setsMatchModelSets() {
        #expect(ModelCatalog.recommended.set == ModelSet.recommended)
        #expect(ModelCatalog.quality.set == ModelSet.quality)
        #expect(ModelCatalog.all.count == 2)
    }

    @Test func everyModelIsPinnedAndHasALicence() {
        for set in ModelCatalog.all {
            #expect(set.models.count == 3)
            for model in set.models {
                guard case .hub(let repo, let file, let sha) = model.source else {
                    Issue.record("\(model.name) is not a hub source")
                    continue
                }
                #expect(repo.split(separator: "/").count == 2)
                #expect(file.hasSuffix(".gguf"))
                #expect(sha?.count == 64, "\(model.name) needs a SHA-256")
                #expect(!model.license.isEmpty)
                #expect(model.sizeBytes > 0)
            }
        }
    }

    @Test func rolesMatchTheSet() {
        for set in ModelCatalog.all {
            #expect(set.models.map(\.role) == [.embedder, .reranker, .generator])
            #expect(set.models[0].source == set.set.embedder)
            #expect(set.models[1].source == set.set.reranker)
            #expect(set.models[2].source == set.set.generator)
        }
    }

    @Test func totalsAndEstimates() {
        let set = ModelCatalog.recommended
        #expect(set.totalSizeBytes == set.models.map(\.sizeBytes).reduce(0, +))
        let largest = set.models.map(\.sizeBytes).max() ?? 0
        #expect(set.lowMemoryEstimateBytes > largest)
        #expect(set.lowMemoryEstimateBytes < set.totalSizeBytes * 2)
    }
}
