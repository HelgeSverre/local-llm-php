<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Generation;

final readonly class GenerationConfig
{
    /**
     * @param list<string> $stopStrings
     * @param list<int> $stopTokens
     * @param list<MediaInput> $mediaInputs
     */
    public function __construct(
        public ?string $prompt = null,
        public int $maxTokens = 128,
        public float $temperature = 0.0,
        public int $topK = 40,
        public float $topP = 0.95,
        public float $minP = 0.05,
        public int $seed = 0xFFFFFFFF,
        public bool $addSpecial = true,
        public bool $parseSpecial = true,
        public bool $removeSpecialOnDetokenize = false,
        public bool $unparseSpecial = true,
        public array $stopStrings = [],
        public array $stopTokens = [],
        public array $mediaInputs = [],
        public ?string $mediaMarker = null,
    ) {
        if ($maxTokens < 0) {
            throw new \InvalidArgumentException('Max tokens must be zero or greater.');
        }

        if ($temperature < 0.0) {
            throw new \InvalidArgumentException('Temperature must be zero or greater.');
        }

        if ($topK < 0) {
            throw new \InvalidArgumentException('Top-k must be zero or greater.');
        }

        if ($topP <= 0.0 || $topP > 1.0) {
            throw new \InvalidArgumentException('Top-p must be greater than zero and at most one.');
        }

        if ($minP < 0.0 || $minP > 1.0) {
            throw new \InvalidArgumentException('Min-p must be between zero and one.');
        }

        foreach ($mediaInputs as $mediaInput) {
            if (!$mediaInput instanceof MediaInput) {
                throw new \InvalidArgumentException('Media inputs must be instances of ' . MediaInput::class . '.');
            }
        }

        if ($mediaMarker !== null && trim($mediaMarker) === '') {
            throw new \InvalidArgumentException('Media marker must not be empty when provided.');
        }
    }
}
