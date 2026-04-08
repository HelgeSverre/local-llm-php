<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\FFI\LlamaBackendConfig;
use HelgeSverre\LocalLlm\Support\RuntimePlatform;
use PHPUnit\Framework\TestCase;
use Psr\Log\LogLevel;

final class LlamaBackendConfigTest extends TestCase
{
    public function testItKeepsNativeLogRoutingEnabledByDefault(): void
    {
        $config = new LlamaBackendConfig('/tmp/' . RuntimePlatform::sharedLibraryBasename());

        self::assertTrue($config->captureNativeLogs);
        self::assertSame(LogLevel::WARNING, $config->nativeLogLevel);
    }
}
