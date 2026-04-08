# Changelog

## Unreleased

- Initial standalone release of `helgesverre/local-llm-php`
- `llama.cpp` FFI backend with model load, tokenization, prompt evaluation, iterative decode, and streaming callbacks
- model-aware chat API with native template support, fallback formatting, and prompt override escape hatch
- PSR-3 logging with native `llama.cpp` log routing
- profiling for prompt evaluation, decode throughput, and wall time
- supported-model catalog and Apple Silicon presets
- benchmark harness against local Ollama
- Docker-based Linux smoke verification and cross-platform build support
