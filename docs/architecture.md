# Architecture And Feasibility

## Objective

This package exists to answer one question: can a native local-LLM backend, driven from PHP through FFI, get materially closer to Ollama-class performance than the current ONNX Runtime path on Apple Silicon?

The implementation is isolated from the ONNX Runtime PHP package on purpose. It is structured as a standalone package with a backend-neutral PHP surface and a backend-specific native bridge.

## Current architecture

### PHP surface

- `BackendInterface`, `ModelInterface`, and `SessionInterface` define the swappable backend contract.
- `GenerationConfig`, `GenerationResult`, `PromptEvaluationResult`, `GenerationProfile`, and `SessionState` provide backend-neutral generation and profiling types.
- `LocalLlm::llamaCpp()` is a convenience entry point for the first backend.
- `Runtime\LlamaRuntime` is the higher-level convenience layer for safer setup, model resolution, and session creation.

### Native bridge

- `src/FFI/LlamaCdef.php` contains the subset of the official `llama.h` C ABI that this package uses.
- `LlamaLibrary` owns the FFI binding and global backend initialization.
- `LlamaModel` owns the `llama_model *` and tokenization helpers.
- `LlamaSession` owns the `llama_context *`, prompt evaluation, generation loop, streaming, profiling, reset, and the current experimental snapshot/restore wrapper.

### Session model

- Model state is explicit. There is no hidden global session cache.
- Prompt evaluation appends to session history.
- Generation can continue from previously evaluated context.
- Full serialized snapshot and restore are wired to the official llama.cpp state APIs, but the normal reuse/reset path is better qualified than the serialized-state path right now.

## Why `llama.cpp` works for this experiment

The upstream `llama.cpp` C API is already shaped for this use case:

- model load and unload
- tokenizer and detokenizer access
- explicit prompt evaluation through `llama_decode` and `llama_encode`
- sampler-chain composition
- performance counters for prompt eval, decode, and sampling
- session state snapshot and restore

That makes PHP FFI viable without introducing a custom C shim for the first backend.

## What worked

- Loading a local GGUF model from PHP through FFI
- Tokenizing and detokenizing through the native vocab
- Prompt evaluation and iterative token generation
- Streaming decoded text through a PHP callback
- Capturing both wall-clock timings and native llama.cpp perf counters
- Reusing in-memory session state across prompt-eval and generation steps
- Comparing the same local model against Ollama with a dedicated harness

## Risky areas and mitigations

### Resource lifecycle

FFI-backed pointers are unforgiving. The package keeps ownership explicit:

- model and session wrappers expose `close()`
- closed handles now fail fast instead of risking use-after-free
- temporary token batches are kept alive across `llama_decode` and `llama_encode`

### Benchmark fairness

Ollama can wrap prompts in its model template and can keep models resident across requests. The benchmark harness therefore defaults to:

- raw Ollama prompts
- `keep_alive=0`

That keeps prompt token counts and load behavior closer to the direct `llama.cpp` path.

### Serialized session state

The package currently exposes a serialized session-state wrapper, but that path still needs more validation on this Metal-backed setup. The benchmarked and verified path is in-memory session reuse plus explicit `reset()`.

### Setup correctness

The package now includes explicit environment inspection and Ollama model-path resolution so common setup failures can be detected before the first native call:

- FFI extension loaded
- `ffi.enable` enabled
- native library present
- likely loader-path mismatch for custom shared-library layouts
- local Ollama model resolving to a GGUF blob path

## MLX feasibility

MLX remains interesting, but it is not the right second backend for phase 1 under the current constraints.

Why:

- the official low-level C surface, `mlx-c`, is tensor-oriented rather than LLM-runtime-oriented
- the official higher-level generation stack is `mlx-lm`, which is Python-based
- building a serious PHP-native MLX backend would therefore require a custom native shim that handles model loading, KV-cache lifecycle, sampling, streaming, and likely tokenizer integration

That is technically possible, but it is a larger project than the `llama.cpp` path and would blur the experiment by mixing backend evaluation with substantial shim design.

Recommendation:

- keep MLX out of the critical path for now
- finish the `llama.cpp` measurements first
- only revisit MLX if `llama.cpp` via FFI is promising enough that a second backend is worth the native-shim cost

## Bottom line

`llama.cpp` is a credible foundation for a real standalone PHP package. The core C ABI is stable enough to support a clean FFI layer, and the initial runtime path already supports the features needed to answer the throughput question honestly.
