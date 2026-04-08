<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Generation;

final readonly class GenerationProfile
{
    public function __construct(
        public int $promptTokens,
        public int $generatedTokens,
        public int $promptEvalDurationUs,
        public int $decodeDurationUs,
        public int $totalDurationUs,
        public float $nativePromptEvalMs,
        public float $nativeDecodeMs,
        public float $nativeSampleMs,
        public int $nativeGraphReuseCount,
    ) {
    }

    public function promptTokensPerSecond(): float
    {
        if ($this->promptEvalDurationUs <= 0 || $this->promptTokens === 0) {
            return 0.0;
        }

        return $this->promptTokens / ($this->promptEvalDurationUs / 1_000_000);
    }

    public function decodeTokensPerSecond(): float
    {
        if ($this->decodeDurationUs <= 0 || $this->generatedTokens === 0) {
            return 0.0;
        }

        return $this->generatedTokens / ($this->decodeDurationUs / 1_000_000);
    }
}
