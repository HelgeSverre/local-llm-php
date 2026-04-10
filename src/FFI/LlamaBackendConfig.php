<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\FFI;

use Psr\Log\LoggerInterface;
use Psr\Log\LogLevel;
use Psr\Log\NullLogger;

final readonly class LlamaBackendConfig
{
    public function __construct(
        public string $libraryPath,
        public bool $initializeBackend = true,
        public bool $loadDynamicBackends = true,
        public LoggerInterface $logger = new NullLogger(),
        public bool $captureNativeLogs = true,
        public string $nativeLogLevel = LogLevel::WARNING,
    ) {}
}
