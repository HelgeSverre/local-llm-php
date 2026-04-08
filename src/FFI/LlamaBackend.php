<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\FFI;

use HelgeSverre\LocalLlm\Backend\BackendInterface;
use HelgeSverre\LocalLlm\Backend\ModelInterface;
use HelgeSverre\LocalLlm\Backend\ModelOptions;
use HelgeSverre\LocalLlm\Support\RuntimePlatform;

final class LlamaBackend implements BackendInterface
{
    private readonly LlamaLibrary $library;

    public function __construct(?LlamaBackendConfig $config = null)
    {
        $this->library = new LlamaLibrary($config ?? new LlamaBackendConfig(self::defaultLibraryPath()));
    }

    public function name(): string
    {
        return 'llama.cpp';
    }

    public function loadModel(ModelOptions $options): ModelInterface
    {
        return new LlamaModel($this->library, $options);
    }

    public static function defaultLibraryPath(): string
    {
        return dirname(__DIR__, 2) . '/var/native/llama.cpp/build/bin/' . RuntimePlatform::sharedLibraryBasename();
    }
}
