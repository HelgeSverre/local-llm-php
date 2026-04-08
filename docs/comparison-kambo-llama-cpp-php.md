# Comparison: `kambo/llama-cpp-php`

This note records a direct comparison against [`kambo-1st/llama-cpp-php`](https://github.com/kambo-1st/llama-cpp-php) at commit `5762c8feba308606f6a7a37fafc21b2e1152811a` from April 24, 2023.

## Outcome

A fair benchmark against the current `local-llm-php` package was **not possible without porting the other package first**.

The blocker is not minor tuning. The `kambo` package targets a much older `llama.cpp` API and model format generation:

- It documents `ggjt-model.bin`, not GGUF.
- It declares Linux-only support in its README.
- It binds removed functions such as `llama_init_from_file`, `llama_eval`, `llama_sample_top_p_top_k`, and `llama_token_to_str`.
- The current Apple Silicon `libllama.dylib` used by this package exports the modern API surface instead, including `llama_model_load_from_file`, `llama_decode`, `llama_sampler_chain_init`, and `llama_chat_apply_template`.

When loaded against this package's current `libllama.dylib`, the other package fails immediately during `FFI::cdef(...)` with:

```text
Failed resolving C function 'llama_init_from_file'
```

That means there is no apples-to-apples benchmark path on the same backend and same model blob without first forking or porting the `kambo` package to the modern `llama.cpp` C API.

## Direct comparison

### `local-llm-php`

- Focus: current `llama.cpp` through the modern C API
- Platform target: macOS Apple Silicon first, with Linux CPU support now validated separately
- Model format: GGUF
- Session model: reusable runtime and session objects
- Features:
  - prompt evaluation
  - iterative decode
  - streaming callbacks
  - model-aware chat templating
  - PSR-3 logging
  - structured profiling
  - supported-model matrix and presets
  - benchmark harness vs Ollama
  - native integration coverage

### `kambo/llama-cpp-php`

- Focus: early experimental PHP binding over an older `llama.cpp`
- Platform target: Linux only, per its README
- Model format: old `ggjt-model.bin` example
- Session model: one low-level context wrapper plus a simple generator API
- Features:
  - prompt tokenize/eval loop
  - token generation
  - optional event dispatch on token output
  - embedding extraction

## Codebase snapshot

Observed locally during this comparison:

- `local-llm-php`
  - `src`: 41 PHP files
  - `tests`: 15 PHP files
  - implementation size: about 2986 lines under `src`
  - tests: `37` passing in the default suite, with additional native integration coverage available locally

- `kambo/llama-cpp-php`
  - `src`: 12 PHP files
  - `tests`: 4 PHP files
  - implementation size: about 651 lines under `src`
  - tests: `11` passing tests, all unit-level

This is not a criticism of the smaller package. It was built against an earlier state of the ecosystem. It is just no longer a suitable production or benchmark peer for a current GGUF-based Apple Silicon package.

## What would be required to benchmark it fairly

To benchmark both packages on the same machine and same native backend, the `kambo` package would first need a real port:

1. Replace the old header and FFI bindings with the modern `llama.cpp` C API.
2. Replace `llama_init_from_file` context loading with model plus context creation.
3. Replace `llama_eval` with `llama_decode` batch-based evaluation.
4. Replace old sampling helpers with the modern sampler chain API.
5. Replace old token-to-string calls with the current token-piece APIs.
6. Add GGUF-based model handling and macOS Apple Silicon runtime support.

At that point it would effectively become a different package, and the benchmark would be comparing two modernized implementations rather than the original library as published.

## Practical conclusion

For current work, the useful comparison is architectural rather than benchmark-based:

- the older package proves that PHP FFI bindings to `llama.cpp` are viable in principle
- it does **not** provide a meaningful current benchmark baseline for GGUF, Apple Silicon, or modern `llama.cpp`
- the relevant external benchmark target remains Ollama on the same machine and model blob, which this package already measures directly
