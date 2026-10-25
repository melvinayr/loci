# Memory on iOS

Running three models on a phone without being terminated.

## Residency

``ModelResidency`` decides how long models stay loaded. `.lowMemory` keeps one model in memory at a time, swapping between the embedder, reranker and generator; `.keepResident` keeps all of them; `.balanced` sits between. Loci estimates a model's footprint from its GGUF header and throws `insufficientMemory` before loading when it would not fit, instead of letting the system terminate the app. Models are also released when the system reports memory pressure.

## Practical advice

- Use `.lowMemory` on phones with 6 GB, and consider `.balanced` above that.
- Expect the first answer after launch to be slower: the models load on demand.
- Call `unloadModels()` when the app moves to the background.
- Use `MemoryUsage.currentBytes` and `peakBytes` to measure your own app.
- Stored files, including the database, WAL and SHM, are created with file protection.

Speed and memory measured on a Mac are in `docs/BENCHMARKS.md`. Measure on the iPhones you target; numbers from a Mac do not transfer.
