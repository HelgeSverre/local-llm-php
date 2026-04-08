# Local LLM PHP

Standalone local `llama.cpp` runtime for PHP via FFI with GGUF support, streaming, and profiling.

This package is intentionally separate from the ONNX Runtime PHP package. The goal is to measure how close an in-process PHP FFI path can get to Ollama-class performance on Apple Silicon when both sides use the same local GGUF model, and to turn that result into a usable standalone package. The primary performance target is still Apple Silicon, but the runtime and build path now also work on Linux with a CPU-first configuration.

## Status

- `llama.cpp` backend: implemented and working on macOS Apple Silicon.
- Linux support: validated in a clean Debian-based Docker build with `libllama.so` and CPU execution.
- Model-aware message API: implemented through `llama.cpp` native chat templates, with generic fallback and full overrides.
- PSR-3 structured logging: implemented.
- Streaming token callbacks: implemented.
- Prompt eval, decode, and end-to-end profiling: implemented.
- Session reset and in-memory context reuse: implemented.
- Full session snapshot and restore: exposed experimentally, but not yet qualified as stable on this Metal setup.
- Benchmark harness against local Ollama: implemented.

## Documentation

- [Architecture and feasibility](docs/architecture.md)
- [Models, installation, and limitations](docs/models.md)
- [Benchmark notes](docs/benchmark-notes.md)
- [Comparison with `kambo/llama-cpp-php`](docs/comparison-kambo-llama-cpp-php.md)
- [Production priorities](docs/roadmap.md)

## Requirements

- PHP 8.3+ with `ext-ffi`
- macOS Apple Silicon or Linux (`x86_64` and `arm64` are the current validated Linux host targets)
- a locally built `libllama.dylib` on macOS or `libllama.so` on Linux
- a local GGUF model
- local Ollama for side-by-side benchmarks on the same host when benchmarking

## Install

As a dependency:

```bash
composer require helgesverre/local-llm-php
```

From a clone of this repository:

```bash
cd /Users/helge/code/local-llm-php
composer install
./scripts/build-llama-cpp.sh
```

The build script downloads upstream `llama.cpp` into `var/native/llama.cpp` and builds the shared library for the current host:

- macOS: `libllama.dylib` with Metal enabled
- Linux: `libllama.so` with CPU execution enabled

The default build embeds a relative runtime search path, so the generated `libllama` usually resolves sibling `ggml` libraries without extra shell setup. If you use a custom build or move the shared libraries elsewhere, set the platform loader path yourself:

- macOS: `DYLD_LIBRARY_PATH`
- Linux: `LD_LIBRARY_PATH`

If you need custom `llama.cpp` CMake flags, pass them through `LLAMA_CPP_CMAKE_ARGS`.

## Minimal usage

```php
<?php

require __DIR__ . '/vendor/autoload.php';

use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\LocalLlm;

$runtime = LocalLlm::llamaCppRuntime(
    modelPath: '/absolute/path/to/model.gguf',
);

$session = $runtime->newSession();

$result = $session->generate(
    new GenerationConfig(
        prompt: 'Write one sentence about local inference.',
        maxTokens: 64,
        temperature: 0.0,
    ),
    static function ($chunk): void {
        echo $chunk->text;
    },
);

printf(
    "\n\nPrompt eval: %.2f tok/s, decode: %.2f tok/s\n",
    $result->profile->promptTokensPerSecond(),
    $result->profile->decodeTokensPerSecond(),
);

$session->close();
$runtime->close();
```

If you want to force CPU execution explicitly, set `gpuLayers: 0`. When omitted, the runtime defaults to `99` on macOS and `0` on Linux.

If the model is already installed in local Ollama, you can skip the explicit GGUF path:

```php
use HelgeSverre\LocalLlm\LocalLlm;

$runtime = LocalLlm::llamaCppRuntimeFromOllama('granite3.3:2b');
$session = $runtime->newSession();
```

If you want to stay on a qualified preset instead of choosing context and batching manually, use the supported-model catalog:

```php
<?php

use HelgeSverre\LocalLlm\LocalLlm;
use HelgeSverre\LocalLlm\Support\AppleSiliconTier;

$profile = LocalLlm::supportedModel('qwen2.5-3b-instruct-q5-k-m');

$runtime = LocalLlm::llamaCppRuntime(
    modelPath: '/absolute/path/to/qwen2.5-3b-instruct-q5_k_m.gguf',
    sessionOptions: $profile->recommendedSessionOptions(AppleSiliconTier::GB16),
    gpuLayers: $profile->recommendedGpuLayers(AppleSiliconTier::GB16),
);
```

## Message API

For the common chat-style use case, prefer `newChatSession()`. By default it uses the model's own `llama.cpp` chat template when one is available. If not, it falls back to a generic deterministic formatter. You can also override the template or bypass formatting completely.

Native chat-template output is now validated before the high-level API trusts it. If the current `llama.cpp` C API produces malformed output for a model template, the package logs a warning and falls back to the deterministic generic formatter instead of sending a corrupted prompt into generation.

```php
<?php

require __DIR__ . '/vendor/autoload.php';

use HelgeSverre\LocalLlm\Chat\ChatMessage;
use HelgeSverre\LocalLlm\Chat\ChatOptions;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\LocalLlm;

$runtime = LocalLlm::llamaCppRuntimeFromOllama('granite3.3:2b');
$chat = $runtime->newChatSession();

$result = $chat->generate(
    [
        ChatMessage::system('You are terse.'),
        ChatMessage::user('Say hello in three words.'),
    ],
    new GenerationConfig(maxTokens: 16, temperature: 0.0),
);

$rawOverride = $chat->generate(
    [ChatMessage::user('ignored by prompt override')],
    new GenerationConfig(maxTokens: 16, temperature: 0.0),
    options: new ChatOptions(promptOverride: "Write exactly three words: hello from php"),
);
```

## Structured logging

The runtime accepts any PSR-3 logger. The package does not force Monolog, but Monolog works well if you want structured JSON or file logging.

```php
<?php

use Monolog\Handler\StreamHandler;
use Monolog\Logger;
use HelgeSverre\LocalLlm\LocalLlm;

$logger = new Logger('local-llm');
$logger->pushHandler(new StreamHandler('php://stderr'));

$runtime = LocalLlm::llamaCppRuntimeFromOllama(
    'granite3.3:2b',
    logger: $logger,
);
```

Logged events include model load, session creation, prompt evaluation, chat formatting, and generation completion.

Native `llama.cpp` logs are also routed into PSR-3 by default, so raw native stderr is quieter than before. If you want more native detail in logs, lower `nativeLogLevel`. If you want raw native stderr back for debugging, disable capture at runtime construction.

The backend also routes the common `llama.cpp` and `ggml` native log callback through PSR-3 by default, which suppresses the usual verbose native stderr output in the common case. Direct `fprintf(stderr, ...)` paths that bypass the callback can still surface in some upstream/native edge cases.

## Setup validation

The package now includes a higher-level runtime wrapper and a small doctor CLI:

```bash
cd /Users/helge/code/local-llm-php
php -d ffi.enable=1 ./bin/doctor --ollama-model granite3.3:2b
```

That checks:

- whether `ext-ffi` is loaded
- whether `ffi.enable` is on
- whether the expected `libllama` shared library exists for the current host
- whether the relevant loader path variable looks sane for sibling `ggml` libraries
- whether an Ollama model can be resolved to a local GGUF blob

If you also want to load the model and print metadata, add `--inspect-model 1`.

If you want the current qualified model matrix from the CLI, add `--list-supported-models 1`.

## Benchmarking

The benchmark harness compares this package against local Ollama using the same local Ollama model and, when possible, the same GGUF blob path resolved from `ollama show --modelfile`.

The harness defaults to:

- raw Ollama prompts, so Ollama does not add its own template around the prompt
- `keep_alive=0`, so each Ollama run includes model load instead of reusing a warm resident model
- warm worker measurements for both this package and Ollama
- warmup runs plus multiple measured runs
- separate prompt-eval, decode, total, and wall-clock reporting
- configurable package-side session knobs so batching and warm-worker tuning can be swept directly

Example:

```bash
cd /Users/helge/code/local-llm-php
php -d ffi.enable=1 ./bin/benchmark suite \
  --ollama-model granite3.3:2b \
  --runs 3 \
  --warmups 1
```

That writes JSON and Markdown reports under `var/results/`.

For a Linux CPU-only benchmark in Docker against a mounted local GGUF blob:

```bash
cd /Users/helge/code/local-llm-php
./scripts/benchmark-linux-docker.sh
```

That uses the local Ollama model blob by default, mounts it into a Debian-based container, and compares:

- package inside Linux container
- host Ollama via `http://host.docker.internal:11434`

Treat that result as a Linux CPU-only deployment preview, not a direct replacement for the Apple Silicon host benchmark. The container path does not use Metal and is therefore much slower on this machine.

Useful package tuning flags:

- `--batch-size`
- `--micro-batch-size`
- `--threads`
- `--batch-threads`
- `--flash-attention 0|1`
- `--offload-kqv 0|1`

## Testing

```bash
cd /Users/helge/code/local-llm-php
./vendor/bin/phpunit --testsuite unit
```

Clean-slate Linux install verification is included as a Docker smoke test:

```bash
cd /Users/helge/code/local-llm-php
./scripts/smoke-linux-docker.sh
```

That builds a fresh Debian-based image, installs Composer dependencies, builds `llama.cpp`, runs the unit suite, runs `bin/doctor`, and performs an FFI load of the generated `libllama.so`.

Integration tests are opt-in because they require a local model and native library:

```bash
export LOCAL_LLM_FFI_RUN_INTEGRATION=1
export LOCAL_LLM_FFI_LLAMA_LIB=/absolute/path/to/libllama.so
export LOCAL_LLM_FFI_MODEL=/absolute/path/to/model.gguf
php -d ffi.enable=1 ./vendor/bin/phpunit --testsuite integration
```

On macOS use your `libllama.dylib` path instead. If your custom build does not embed a relative runtime search path, also export `DYLD_LIBRARY_PATH` or `LD_LIBRARY_PATH` accordingly.

The integration suite now covers:

- chat-session formatting sanity on real models
- stream chunk reconstruction versus generated final text
- reset and deterministic reuse
- long-prompt chunking with small `n_batch`
- clean failure paths for missing prompt state and invalid model paths

## Limitations

- The strongest benchmark and tuning data is still on Apple Silicon. Linux is currently a validated build/runtime path, not yet a benchmarked parity target.
- The package is intentionally focused on `llama.cpp`; no second backend is planned right now.
- Linux support is currently CPU-first. If you need GPU acceleration on Linux, that is a separate qualification track.
- Full serialized session-state export/import still needs more validation under the current `llama.cpp` Metal path.
- The package now suppresses the common `llama.cpp` and `ggml` callback-driven native stderr path by routing it through PSR-3, but some direct upstream `fprintf(stderr, ...)` paths may still escape.
- Cold process startup still includes PHP startup, FFI binding setup, and model load; a persistent worker model will amortize that better than short-lived CLI runs.
- Vendor dependencies and native build outputs are intentionally ignored by Git and are generated locally.
