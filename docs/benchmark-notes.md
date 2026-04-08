# Benchmark Notes

Report source:

- JSON: `/Users/helge/code/local-llm-php/var/results/benchmark-20260408-124102.json`
- Markdown: `/Users/helge/code/local-llm-php/var/results/benchmark-20260408-124102.md`

Environment:

- machine: Apple Silicon, `Darwin 24.6.0 arm64`
- PHP: `8.5.4`
- package backend: upstream `llama.cpp` C API through PHP FFI
- comparison target: local Ollama `granite3.3:2b`
- model blob: the local GGUF used by that Ollama model
- Ollama request mode: `raw=true`
- package session options: `contextTokens=4096`, `batchSize=1024`, `microBatchSize=1024`, `flashAttention=true`, `offloadKqv=true`
- runs: `1` warmup + `2` measured in the warm-worker suite

## Median results

### Short prompt

- package cold prompt eval: `11.938 ms`
- package warm prompt eval: `2.087 ms`
- Ollama cold prompt eval: `126.507 ms`
- Ollama warm prompt eval: `69.645 ms`
- package cold decode: `101.360 tok/s`
- package warm decode: `104.692 tok/s`
- Ollama cold decode: `119.710 tok/s`
- Ollama warm decode: `116.501 tok/s`
- package cold wall: `797.368 ms`
- package warm wall: `480.013 ms`
- Ollama cold wall: `1063.693 ms`
- Ollama warm wall: `822.119 ms`

### Long prompt

- package cold prompt eval: `696.728 ms`
- package warm prompt eval: `673.867 ms`
- Ollama cold prompt eval: `1101.757 ms`
- Ollama warm prompt eval: `1142.481 ms`
- package cold decode: `67.287 tok/s`
- package warm decode: `76.792 tok/s`
- Ollama cold decode: `93.404 tok/s`
- Ollama warm decode: `95.625 tok/s`
- package cold wall: `2456.615 ms`
- package warm wall: `1929.080 ms`
- Ollama cold wall: `2641.539 ms`
- Ollama warm wall: `2687.354 ms`

## Interpretation

- The direct PHP FFI path still wins clearly on cold prompt evaluation and overall cold wall time.
- After the latest decode-path cleanup, the short-prompt warm case moved closer to Ollama, but the long-prompt numbers are still noisy at this sample size.
- Warm-worker mode materially improves package wall time on the short case because model load and runtime setup disappear.
- Decode throughput remains the main remaining gap to Ollama in both cold and warm runs.
- The warm long-prompt comparison is still not a pure decode comparison because daemon-style reuse can materially distort the Ollama side.
- The next useful tuning work is package-side warm-worker service mode, sampler-path overhead trimming, and measured sweeps of `batchSize`, `microBatchSize`, and thread settings.

## Important caveats

- Package warm-worker measurements reuse the loaded model and process, but still reset the session before each run.
- The warm Ollama long-prompt result should not be over-interpreted because repeated identical prompts can benefit from reuse inside the resident service.
- Full serialized session-state export/import still needs more validation and was not part of the benchmarked path.
