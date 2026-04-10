<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\Generation\MediaInput;
use InvalidArgumentException;
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

    public function testItRejectsInvalidMediaInputs(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $invalidMediaInputs = ['not-a-media-input'];
        /** @var mixed $invalidMediaInputs */
        new GenerationConfig(mediaInputs: $invalidMediaInputs);
    }

    public function testItAcceptsTypedMediaInputs(): void
    {
        $config = new GenerationConfig(mediaInputs: [MediaInput::fromFile('/tmp/example.png')]);

        self::assertCount(1, $config->mediaInputs);
    }
}
