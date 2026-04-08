<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\Generation\GenerationProfile;
use PHPUnit\Framework\TestCase;

final class GenerationProfileTest extends TestCase
{
    public function testItComputesPromptAndDecodeRates(): void
    {
        $profile = new GenerationProfile(
            promptTokens: 100,
            generatedTokens: 50,
            promptEvalDurationUs: 2_000_000,
            decodeDurationUs: 1_000_000,
            totalDurationUs: 3_000_000,
            nativePromptEvalMs: 0.0,
            nativeDecodeMs: 0.0,
            nativeSampleMs: 0.0,
            nativeGraphReuseCount: 0,
        );

        self::assertSame(50.0, $profile->promptTokensPerSecond());
        self::assertSame(50.0, $profile->decodeTokensPerSecond());
    }
}
