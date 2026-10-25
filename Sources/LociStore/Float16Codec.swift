import Accelerate

/// Float32 ↔ IEEE half-precision conversion through Accelerate. Swift's `Float16` is
/// unavailable on Intel Macs, so vectors are handled as raw `UInt16` bit patterns.
enum Float16Codec {
    /// Rounds to the nearest half-precision value.
    static func encode(_ values: [Float]) -> [UInt16] {
        guard !values.isEmpty else { return [] }
        var source = values
        var output = [UInt16](repeating: 0, count: values.count)
        source.withUnsafeMutableBytes { src in
            output.withUnsafeMutableBytes { dst in
                var from = vImage_Buffer(data: src.baseAddress, height: 1, width: vImagePixelCount(values.count), rowBytes: src.count)
                var to = vImage_Buffer(data: dst.baseAddress, height: 1, width: vImagePixelCount(values.count), rowBytes: dst.count)
                vImageConvert_PlanarFtoPlanar16F(&from, &to, vImage_Flags(kvImageNoFlags))
            }
        }
        return output
    }

    /// Converts `count` half values starting at `source` into `destination`.
    static func decode(_ source: UnsafePointer<UInt16>, count: Int, into destination: UnsafeMutablePointer<Float>) {
        var from = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: source), height: 1, width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<UInt16>.stride)
        var to = vImage_Buffer(
            data: destination, height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float>.stride)
        vImageConvert_Planar16FtoPlanarF(&from, &to, vImage_Flags(kvImageNoFlags))
    }

    static func decode(_ values: [UInt16]) -> [Float] {
        guard !values.isEmpty else { return [] }
        var output = [Float](repeating: 0, count: values.count)
        values.withUnsafeBufferPointer { src in
            output.withUnsafeMutableBufferPointer { dst in
                if let s = src.baseAddress, let d = dst.baseAddress { decode(s, count: values.count, into: d) }
            }
        }
        return output
    }
}
