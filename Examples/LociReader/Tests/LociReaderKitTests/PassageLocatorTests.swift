import Foundation
import SwiftUI
import Testing
@testable import LociReaderKit

@Suite("Passage locator")
struct PassageLocatorTests {
    private func located(_ excerpt: String, in page: String) -> String? {
        PassageLocator.locate(excerpt: excerpt, inPageText: page).map { (page as NSString).substring(with: $0) }
    }

    @Test func findsAPassageAndReturnsTheTextExactlyAsThePageHasIt() {
        let page = "Results\nParticipant flow. Of 531 people screened, 412 were randomised: 138 to 20 mg.\nPrimary outcome. Fell by 3.8."
        #expect(located("Of 531 people screened, 412 were randomised: 138 to 20 mg.", in: page) == "Of 531 people screened, 412 were randomised: 138 to 20 mg.")
    }

    @Test func ignoresDifferencesInLineBreaksSpacesAndCase() {
        let page = "The trial  enrolled\n412   participants\nacross nine sites."
        #expect(located("the trial enrolled 412 participants across nine sites.", in: page) == page)
        #expect(located("412 participants", in: page) == "412   participants")
    }

    @Test func seesThroughLigaturesAndCompatibilityForms() {
        let page = "The ﬁnal results show a signiﬁcant ﬂuctuation in ﬁsh biomass."
        #expect(located("final results show a significant fluctuation", in: page) == "ﬁnal results show a signiﬁcant ﬂuctuation")
        #expect(located("fish biomass", in: page) == "ﬁsh biomass")
    }

    @Test func mapsBackToUTF16OffsetsAcrossEmojiAndAccents() {
        let page = "Intro 😀 café notes. The key finding is here. Outro."
        let range = PassageLocator.locate(excerpt: "The key finding is here.", inPageText: page)
        #expect(range.map { (page as NSString).substring(with: $0) } == "The key finding is here.")
        #expect(located("café notes", in: page) == "café notes")
    }

    @Test func ignoresPunctuationAndHyphenationDifferences() {
        let page = "A deep-brown crust means the loaf is done; cool it for one hour."
        #expect(located("A deep brown crust means the loaf is done, cool it for one hour", in: page) == page.dropLast().description)
    }

    @Test func returnsNilWhenThePassageIsNotOnThePage() {
        #expect(located("Something else entirely", in: "A page about lighthouses and granite towers.") == nil)
        #expect(located("", in: "text") == nil)
        #expect(located("text", in: "") == nil)
        #expect(located("!!! ???", in: "text") == nil, "an excerpt with no letters or digits cannot be located")
    }

    @Test func fallsBackToTheStartAndEndWhenTheMiddleDiffers() {
        let start = "The Cape Marrow lighthouse was built in 1874 on the northern headland of the bay"
        let end = "and the total cost was sixty one thousand pounds roughly double the estimate"
        let excerpt = start + " MIDDLE WORDS THAT THE PDF DOES NOT HAVE " + end
        let page = "Heading. " + start + " different middle text here " + end + ". Footer."
        let found = located(excerpt, in: page)
        #expect(found?.hasPrefix("The Cape Marrow") == true && found?.hasSuffix("double the estimate") == true)
    }

    @Test func theFallbackRefusesWhenTheEndsAreFarApartOrOutOfOrder() {
        let head = String(repeating: "alpha beta gamma delta ", count: 5)
        let tail = String(repeating: "omega sigma kappa theta ", count: 5)
        let excerpt = head + "unmatched middle section " + tail
        #expect(located(excerpt, in: tail + String(repeating: "filler words here ", count: 80) + head) == nil, "tail before head")
        #expect(located(excerpt, in: head + String(repeating: "filler words here ", count: 200) + tail) == nil, "far apart")
    }

    @Test func choosesTheFirstOccurrenceWhenAPassageRepeats() {
        let page = "Repeat me. Gap. Repeat me. End."
        let range = PassageLocator.locate(excerpt: "Repeat me.", inPageText: page)
        #expect(range?.location == 0)
    }

    @Test func handlesALongPageQuickly() {
        let page = String(repeating: "Lorem ipsum dolor sit amet consectetur. ", count: 5_000) + "The needle sentence is here." + String(repeating: " More text.", count: 5_000)
        let start = ContinuousClock.now
        let found = located("The needle sentence is here.", in: page)
        #expect(found == "The needle sentence is here.")
        #expect(ContinuousClock.now - start < .seconds(2))
    }
}

@Suite("Highlighted text")
struct HighlightedTextTests {
    @Test func highlightsExactlyTheCitedScalarRange() {
        let text = "Alpha beta.\n\nGamma 😀 delta. Epsilon."
        let scalars = Array(text.unicodeScalars)
        let start = scalars.firstIndex(of: "G") ?? 0
        let range = start..<(start + "Gamma 😀 delta.".unicodeScalars.count)
        let result = HighlightedText.make(documentText: text, scalarRange: range)
        #expect(result.highlighted == "Gamma 😀 delta.")
        var marked = ""
        for run in result.text.runs where run.backgroundColor != nil { marked += String(result.text[run.range].characters) }
        #expect(marked == "Gamma 😀 delta.")
        #expect(String(result.text.characters) == text, "the text itself is unchanged")
    }

    @Test func aRangeThatDoesNotFitHighlightsNothingAndDoesNotCrash() {
        for range in [5..<500, -1..<3, 4..<4, 100..<110] {
            let result = HighlightedText.make(documentText: "short text", scalarRange: range)
            #expect(result.highlighted.isEmpty && String(result.text.characters) == "short text")
            #expect(result.text.runs.allSatisfy { $0.backgroundColor == nil })
        }
    }
}
