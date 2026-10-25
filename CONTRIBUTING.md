# Contributing to Loci

Thanks for helping make on-device answers trustworthy.

## Setup

```bash
git clone https://github.com/melvinayr/loci.git
cd Loci
./Scripts/fetch-test-models.sh
swift build
swift test
```

Requires Xcode 16+ on macOS (Apple Silicon recommended).

## Before you start

- Open an issue for anything bigger than a bug fix.
- Read `docs/SPEC.md` and `docs/ARCHITECTURE.md`. Layering is enforced in CI: llama.cpp only in `LociLlama`, SQLite only in `LociStore`, networking only in `LociHub`.
- Public API changes update `docs/SPEC.md` in the same PR.

## Pull request checklist

- [ ] `swift build` / `swift test` pass with zero warnings
- [ ] Citation invariant tests pass
- [ ] If you changed parsing, chunking, retrieval, prompts or verification: attach `loci-cli eval` output (before vs. after)
- [ ] Doc comments for new public symbols
- [ ] `CHANGELOG.md` updated under **Unreleased**

## Style

- [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/)
- async/await; actors for stateful components
- No `fatalError`, `try!` or force unwraps on user-reachable paths
- `os.Logger` only — and never log document text, questions or answers

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `perf:`, `eval:`, `refactor:`, `test:`, `chore:`.

## Adding a document format

Implement a parser in `Sources/LociIngest/Parsers` that returns normalised text plus a page/section map, add fixture files, and extend the invariant tests.

## Contributing evaluation data

Only documents with a licence that allows redistribution. Include the source and licence in `Tests/Fixtures/eval-small/SOURCES.md`.

## Release process (maintainers)

1. Move Unreleased entries in `CHANGELOG.md` to the new version.
2. Run the full eval and `loci-cli bench` on a physical iPhone and Mac; update `docs/BENCHMARKS.md`.
3. Tag `vX.Y.Z`, publish a GitHub Release from the changelog.
4. Confirm Swift Package Index rebuilt the docs.

## Code of Conduct

[Contributor Covenant](CODE_OF_CONDUCT.md).
