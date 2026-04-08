<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use InvalidArgumentException;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use PHPUnit\Framework\TestCase;

final class GenerationConfigTest extends TestCase
{
    public function testItRejectsNegativeMaxTokens(): void
    {
        $this->expectException(InvalidArgumentException::class);

        new GenerationConfig(maxTokens: -1);
    }

    public function testItRejectsInvalidTopP(): void
    {
        $this->expectException(InvalidArgumentException::class);

        new GenerationConfig(topP: 1.1);
    }
}
