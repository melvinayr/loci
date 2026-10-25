import Testing
@testable import LociStore

@Suite("SQLite capabilities")
struct SQLiteCapabilityTests {
    @Test func systemSQLiteHasFTS5() {
        #expect(SQLiteCapabilities.hasFTS5(), "FTS5 missing; vendor the amalgamation")
    }

    @Test func compileOptionsListFTS5() {
        #expect(SQLiteCapabilities.compileOptions().contains("ENABLE_FTS5"))
    }
}
