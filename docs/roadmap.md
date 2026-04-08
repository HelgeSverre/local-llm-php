# Production Priorities

## Immediate priorities

1. Stabilize the current `llama.cpp` backend surface.
2. Lock down setup validation and operator ergonomics.
3. Expand testing around lifecycle, long prompts, and error cases.
4. Qualify the new model-aware message API across supported model families.
5. Introduce a long-lived worker or server mode so model load is amortized in real applications.
6. Keep the documented supported-model matrix and the typed preset catalog aligned.

## What "production grade" means here

### Robustness

- fail before native work when FFI or library setup is wrong
- validate option invariants consistently
- make resource ownership explicit and idempotent
- separate experimental APIs from qualified APIs

### Correctness

- deterministic prompt handling for supported model families
- long-prompt handling that respects `n_batch`
- stable stop conditions and token accounting
- accurate profiling semantics that distinguish prompt eval, decode, and load

### Testing

- fast unit tests for validation, setup parsing, and non-native helpers
- focused native integration tests for load, tokenize, generate, stream, reset, and long-prompt chunking
- benchmark regression tests for prompt/decode throughput envelopes
- at least one smoke path that runs in a fresh process to catch native startup regressions

## Recommended next engineering steps

1. Promote `LlamaRuntime` and `bin/doctor` as the default setup path.
2. Add streaming integration tests that assert chunk ordering and final text reconstruction.
3. Add a native smoke script for CI that loads a tiny local test model when available.
4. Split stable APIs from experimental ones, especially serialized state export/import.
5. Route more native/runtime events through PSR-3 logging and reduce dependence on raw native stderr.
6. Add warm-worker benchmarking to complement cold-process benchmarking.
7. Add configurable load/session presets for Apple Silicon memory tiers.
8. Add stronger compatibility docs around quantization, context size, and Metal memory trade-offs.
9. Add prompt-cache style session reuse for repeated prompts once a safe state-reuse strategy is qualified.
10. Keep the package focused on `llama.cpp`; no MLX work is planned.
11. Add warm-worker tuning sweeps and track decode-throughput regressions against pinned presets.
