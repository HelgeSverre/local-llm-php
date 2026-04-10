<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\Backend\ModelOptions;
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;

final class ModelOptionsTest extends TestCase
{
    public function testItRejectsAnEmptyModelPath(): void
    {
        $this->expectException(InvalidArgumentException::class);

        new ModelOptions('');
    }

    public function testItRejectsInvalidGpuLayers(): void
    {
        $this->expectException(InvalidArgumentException::class);

        new ModelOptions('/tmp/model.gguf', gpuLayers: -2);
    }

    public function testItRejectsAnEmptyMultimodalProjectorPath(): void
    {
        $this->expectException(InvalidArgumentException::class);

        new ModelOptions('/tmp/model.gguf', multimodalProjectorPath: '   ');
    }

    public function testItAcceptsMultimodalProjectorConfiguration(): void
    {
        $options = new ModelOptions(
            '/tmp/model.gguf',
            multimodalProjectorPath: '/tmp/mmproj.gguf',
            multimodalProjectorUseGpu: false,
        );

        self::assertSame('/tmp/mmproj.gguf', $options->multimodalProjectorPath);
        self::assertFalse($options->multimodalProjectorUseGpu);
    }
}
