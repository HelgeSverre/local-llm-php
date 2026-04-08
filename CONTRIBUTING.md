# Contributing

## Local development

```bash
cd /Users/helge/code/local-llm-php
composer install
./scripts/build-llama-cpp.sh
```

## Test commands

Unit tests:

```bash
./vendor/bin/phpunit --testsuite unit
```

Optional native integration tests:

```bash
export LOCAL_LLM_FFI_RUN_INTEGRATION=1
export LOCAL_LLM_FFI_LLAMA_LIB=/absolute/path/to/libllama.dylib
export LOCAL_LLM_FFI_MODEL=/absolute/path/to/model.gguf
php -d ffi.enable=1 ./vendor/bin/phpunit --testsuite integration
```

On Linux, point `LOCAL_LLM_FFI_LLAMA_LIB` at `libllama.so` instead.

Clean-slate Linux smoke test:

```bash
./scripts/smoke-linux-docker.sh
```

## Build notes

- macOS builds enable Metal by default.
- Linux builds use a CPU-first configuration by default.
- If you need custom upstream build flags, set `LLAMA_CPP_CMAKE_ARGS`.

## Scope

- `llama.cpp` is the only supported backend.
- Avoid adding Python dependencies to the runtime package.
- Keep FFI lifecycle changes covered by focused tests.
