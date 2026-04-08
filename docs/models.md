# Models

## What this package can load today

The implemented backend is `llama.cpp`, so the model format is currently:

- GGUF files that upstream `llama.cpp` can load

Best-supported use cases right now:

- text-generation and instruct/chat models
- local quantized GGUF models that fit Apple Silicon memory comfortably
- smaller GGUF models on Linux VPS hosts when running CPU-only
- single-model, single-session generation workloads
- model-aware message formatting through the model's own `llama.cpp` chat template when that native output is sane, with deterministic fallback otherwise

## Current practical limitations

### Package limitations

- `llama.cpp` is the only implemented backend.
- serialized session snapshot/restore is still experimental on the current Metal setup
- no grammar-constrained decoding API yet
- no LoRA adapter management API yet
- no multimodal abstraction at the PHP layer yet
- no robust persistent-worker/server mode yet

### Model-selection limitations

- not every GGUF on the internet is well-formed or current with upstream `llama.cpp`
- prompt formatting quality varies a lot by model family
- the current `llama.cpp` C chat-template helper is not a full general-purpose Jinja path, so some model templates need a safe fallback
- large context settings can consume substantial unified memory on Apple Silicon
- quantization choice affects both decode speed and response quality
- Linux CPU inference is much more sensitive to model size than Apple Silicon Metal offload

## Recommended acquisition paths

### Easiest path: install through Ollama, then reuse the local GGUF

This is the smoothest operator experience for now because Ollama handles pull and storage.

Example:

```bash
ollama pull granite3.3:2b
```

Then either:

- benchmark directly against that Ollama model, or
- let the package resolve the local GGUF blob via `LlamaRuntime::fromOllamaModel(...)`

### Direct GGUF download

You can also download GGUF files directly and point the package at the file path.

This path is useful when:

- you want a specific quantization not packaged in Ollama
- you want to pin an exact artifact
- you want to avoid Ollama as the acquisition mechanism

Examples from current primary-source model cards:

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q5_k_m.gguf --local-dir . --local-dir-use-symlinks False
```

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct-GGUF --include "qwen2.5-coder-14b-instruct-q5_k_m*.gguf" --local-dir . --local-dir-use-symlinks False
```

## Suggested starter models

These are sensible starter profiles for Apple Silicon experiments:

- one small instruct model around 2B to 4B parameters for fast iteration
- one second family with a different tokenizer/template behavior to avoid overfitting the package to one architecture

For the current worktree, `granite3.3:2b` is already benchmarked and easy to reuse locally.

For Linux CPU-only deployments, start smaller unless you already know the host is strong enough:

- `1B` to `3B` instruct models are the sensible starting range
- prefer `Q4_K_M` or `Q5_K_M`
- leave `gpuLayers` unset or set `gpuLayers=0`

## Supported model matrix

The table below is the currently qualified path for real use on this package, not a blanket claim about every model that `llama.cpp` can technically open.

| Profile | Source | Artifact | Recommended quantization | Apple Silicon tier | Default package preset |
| --- | --- | --- | --- | --- | --- |
| Fast general chat | [Ollama `granite3.3:2b`](https://ollama.com/library/granite3.3) | local Ollama-managed GGUF blob resolved via `ollama show --modelfile` | use the packaged Ollama quantization | 16 GB+ | `contextTokens=4096`, `batchSize=512`, `microBatchSize=512`, `gpuLayers=99` |
| Small general instruct | [Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) | `qwen2.5-3b-instruct-q5_k_m.gguf` | `Q5_K_M` | 16 GB+ | `contextTokens=4096`, `batchSize=512`, `microBatchSize=512`, `gpuLayers=99` |
| Large coding instruct | [Qwen/Qwen2.5-Coder-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF) | `qwen2.5-coder-14b-instruct-q5_k_m-00001-of-00002.gguf` plus `...00002-of-00002.gguf` | `Q5_K_M` | 64 GB | `contextTokens=8192`, `batchSize=1024`, `microBatchSize=512`, `gpuLayers=99` |
| Ultra-small IBM family trial | [ibm-granite/granite-4.0-1b-GGUF](https://huggingface.co/ibm-granite/granite-4.0-1b-GGUF) | `granite-4.0-1b-Q4_0.gguf` | `Q4_0` | 16 GB | `contextTokens=4096`, `batchSize=512`, `microBatchSize=256`, `gpuLayers=99` |

Notes:

- The Granite `2B` profile is currently qualified through local Ollama because that is the artifact we benchmarked directly.
- `Q5_K_M` is the current default recommendation when the model still fits comfortably in unified memory.
- For 16 GB machines, stay conservative on both model size and context size until you have measurements for your own workload.
- For 64 GB machines, the 14B coder profile is a practical upper bound for the package's current focus.
- On Linux CPU-only hosts, treat the first three profiles as the realistic path until you have host-specific throughput numbers.
- The same qualified matrix is exposed in code through `LocalLlm::supportedModels()` and `LocalLlm::supportedModel($id)`.

## Installation and discovery flows

### From an existing GGUF path

```php
use HelgeSverre\LocalLlm\LocalLlm;

$runtime = LocalLlm::llamaCppRuntime('/absolute/path/to/model.gguf');
$chat = $runtime->newChatSession();
```

### From a locally installed Ollama model

```php
use HelgeSverre\LocalLlm\LocalLlm;

$runtime = LocalLlm::llamaCppRuntimeFromOllama('granite3.3:2b');
$chat = $runtime->newChatSession();
```

### From a qualified model profile

```php
use HelgeSverre\LocalLlm\LocalLlm;
use HelgeSverre\LocalLlm\Support\AppleSiliconTier;

$profile = LocalLlm::supportedModel('qwen2.5-3b-instruct-q5-k-m');

$runtime = LocalLlm::llamaCppRuntime(
    modelPath: '/absolute/path/to/qwen2.5-3b-instruct-q5_k_m.gguf',
    sessionOptions: $profile->recommendedSessionOptions(AppleSiliconTier::GB16),
    gpuLayers: $profile->recommendedGpuLayers(AppleSiliconTier::GB16),
);
```

### Resolve and inspect with the doctor CLI

```bash
cd /Users/helge/code/local-llm-php
php -d ffi.enable=1 ./bin/doctor --ollama-model granite3.3:2b
```

Add `--inspect-model 1` if you want the doctor to load the model and print metadata too.

## Production guidance

- prefer pinned GGUF artifacts, not floating names, once you move beyond experimentation
- record model family, quantization, context, and backend settings together in deployment config
- prefer the typed supported-model catalog in code for qualified profiles instead of copying raw session numbers into multiple call sites
- validate prompt formatting per model family before assuming correctness
- benchmark both cold and warm runs for the exact model you plan to deploy
- set `gpuLayers=0` if you explicitly want CPU-only fallback instead of Metal offload
- on Linux, `gpuLayers` defaults to `0` if you do not set it
- treat the supported matrix above as the qualified path until more models are benchmarked and tested under this package
