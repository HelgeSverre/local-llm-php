<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\Backend\BackendException;
use HelgeSverre\LocalLlm\Runtime\LlamaRuntime;
use HelgeSverre\LocalLlm\Support\RuntimePlatform;
use PHPUnit\Framework\TestCase;

final class LlamaRuntimeTest extends TestCase
{
    public function testItFailsBeforeNativeExecutionWhenTheEnvironmentIsNotReady(): void
    {
        $this->expectException(BackendException::class);
        $this->expectExceptionMessage('Environment is not ready for llama.cpp FFI');

        LlamaRuntime::fromModelPath(
            modelPath: '/tmp/nonexistent-model.gguf',
            libraryPath: '/tmp/nonexistent-' . RuntimePlatform::sharedLibraryBasename(),
        );
    }
}
