import LociReaderKit
import SwiftUI

struct LociReaderApp: App {
    var body: some Scene {
        WindowGroup { ReaderRootView().frame(minWidth: 900, minHeight: 560) }
    }
}

LociReaderApp.main()
