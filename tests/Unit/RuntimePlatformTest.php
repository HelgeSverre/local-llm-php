<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\Support\RuntimePlatform;
use PHPUnit\Framework\TestCase;

final class RuntimePlatformTest extends TestCase
{
    public function testItUsesTheExpectedMacLibraryName(): void
    {
        self::assertSame('libllama.dylib', RuntimePlatform::sharedLibraryBasename('Darwin'));
        self::assertSame('DYLD_LIBRARY_PATH', RuntimePlatform::loaderPathEnvironmentVariable('Darwin'));
        self::assertSame(99, RuntimePlatform::defaultGpuLayers('Darwin'));
    }

    public function testItUsesTheExpectedLinuxLibraryName(): void
    {
        self::assertSame('libllama.so', RuntimePlatform::sharedLibraryBasename('Linux'));
        self::assertSame('LD_LIBRARY_PATH', RuntimePlatform::loaderPathEnvironmentVariable('Linux'));
        self::assertSame(0, RuntimePlatform::defaultGpuLayers('Linux'));
    }
}
