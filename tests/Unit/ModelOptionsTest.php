<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use InvalidArgumentException;
use HelgeSverre\LocalLlm\Backend\ModelOptions;
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
}
