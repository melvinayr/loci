#!/usr/bin/env swift
// Renders a text file into a PDF, one page per form-feed-separated section.
// usage: swift Scripts/make-eval-pdf.swift pages.txt output.pdf
import CoreGraphics
import CoreText
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 3, let source = try? String(contentsOfFile: arguments[1], encoding: .utf8) else {
    FileHandle.standardError.write(Data("usage: make-eval-pdf.swift pages.txt output.pdf\n".utf8))
    exit(1)
}
var box = CGRect(x: 0, y: 0, width: 612, height: 792)
guard let context = CGContext(URL(fileURLWithPath: arguments[2]) as CFURL, mediaBox: &box, nil) else { exit(1) }

let font = CTFontCreateWithName("Helvetica" as CFString, 11, nil)
for section in source.components(separatedBy: "\u{0C}") {
    let text = NSAttributedString(
        string: section.trimmingCharacters(in: .whitespacesAndNewlines),
        attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font])
    context.beginPDFPage(nil)
    let frame = CTFramesetterCreateFrame(
        CTFramesetterCreateWithAttributedString(text), CFRange(location: 0, length: 0),
        CGPath(rect: box.insetBy(dx: 64, dy: 64), transform: nil), nil)
    CTFrameDraw(frame, context)
    context.endPDFPage()
}
context.closePDF()
