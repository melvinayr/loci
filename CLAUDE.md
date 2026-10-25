# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

Loci is a Swift package for grounded, cited, on-device RAG on Apple platforms, built on llama.cpp. Read before any milestone:
- `docs/SPEC.md` — the authoritative design and public API
- `docs/ROADMAP.md` — milestone order and scope
- `docs/ARCHITECTURE.md` — layering and data flow
- `docs/EVALUATION.md` — how quality is measured

## Golden rules

1. **The spec is the contract.** Public API changes must update `docs/SPEC.md` in the same change. If spec and code disagree, stop and ask.
2. **Layering is enforced.** Only `LociLlama` imports llama.cpp. Only `LociStore` touches SQLite. Only `LociHub` uses networking. Dependencies point downward per `docs/ARCHITECTURE.md`.
3. **Citations must be exact.** Never compute offsets on un-normalised text. Any change to parsing, normalisation or chunking must keep the invariant tests passing: every `Citation.range` slices to exactly `Citation.excerpt`, and every PDF citation's page contains its range.
4. **Privacy.** Never log document text, queries or answers. Use `os.Logger` with IDs, counts and timings only. Apply file protection to every file the store creates (DB, WAL, SHM).
5. **Swift 6 strict concurrency, zero warnings.** Every public type `Sendable`. No `@unchecked Sendable` without a comment proving safety.
6. **No crashes.** No `fatalError`, `try!`, force unwraps or `precondition` on user-reachable paths. Convert llama.cpp and SQLite failures into `LociError`.
7. **Every C resource has exactly one owner** that frees it in `deinit`.
8. **Quality is measured, not claimed.** Changes to chunking, retrieval, prompts or verification require a `loci-cli eval` report in the PR description. Never hand-write numbers into docs.
9. **Docs ship with code.** `///` comments on public symbols, README/DocC updates for behaviour changes, a `CHANGELOG.md` entry under Unreleased.

## Commands

```bash
swift build
swift test                                     # unit + integration
swift test --filter LociTests                  # unit only, no models
./Scripts/fetch-test-models.sh                 # tiny GGUF fixtures
./Scripts/update-llama.sh b1234                # bump pinned llama.cpp + checksum
swift run loci-cli index ./Tests/Fixtures/eval-small/documents --kb fixture
swift run loci-cli ask --kb fixture "question"
swift run loci-cli eval --dataset Tests/Fixtures/eval-small --models fixture --out report.md
swift run loci-cli bench --models recommended
swift package generate-documentation --target Loci
```

## Working with llama.cpp

- Check function signatures in the pinned XCFramework headers (`.build/artifacts/**/llama.h`), not from memory — the API changes often.
- Use current APIs (`llama_model_load_from_file`, `llama_init_from_model`, vocab-based token functions, memory APIs).
- Embedding pooling type and rank pooling come from model metadata / context params; verify against the headers.
- Token → text always goes through the UTF-8 buffer.

## Working with SQLite

- Prepared statements cached per connection and finalised in `deinit`.
- One writer, WAL mode, every document ingested in a single transaction.
- Schema changes bump `schema_version` and include a migration test.

## Testing conventions

- Swift Testing (`@Test`, `#expect`).
- Integration tests: CPU only (`gpuLayers: .none`), `temperature: 0`, fixed seed, tiny fixture models.
- Parser tests use files in `Tests/Fixtures/documents`.
- Never commit `*.gguf`.

## Definition of done

- [ ] `swift build` / `swift test` pass, zero warnings
- [ ] Public API matches `docs/SPEC.md`
- [ ] Invariant tests pass
- [ ] Eval report attached if retrieval/prompt/verification changed, no regression beyond CI margins
- [ ] Doc comments, README/DocC, CHANGELOG updated

## Do not

- Add third-party Swift dependencies to `Loci` or its internal targets.
- Vendor llama.cpp source or change how it is consumed without discussion.
- Add networking outside `LociHub`.
- Change the grounding prompt without bumping `PromptVersion`.
