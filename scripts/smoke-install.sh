#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT_DIR"

composer install --no-interaction --prefer-dist
./scripts/build-llama-cpp.sh --refresh
php -d ffi.enable=1 ./vendor/bin/phpunit --testsuite unit
php -d ffi.enable=1 ./bin/doctor
php -d ffi.enable=1 <<'PHP'
<?php

require getcwd() . '/vendor/autoload.php';

use HelgeSverre\LocalLlm\FFI\LlamaBackend;
use HelgeSverre\LocalLlm\FFI\LlamaBackendConfig;
use HelgeSverre\LocalLlm\LocalLlm;

$report = LocalLlm::inspectLlamaCppEnvironment();
if ($report->hasBlockingIssues()) {
    foreach ($report->blockingIssues() as $issue) {
        fwrite(STDERR, $issue->message . PHP_EOL);
    }
    exit(1);
}

$backend = new LlamaBackend(new LlamaBackendConfig(LlamaBackend::defaultLibraryPath()));
echo $backend->name(), PHP_EOL;
PHP
