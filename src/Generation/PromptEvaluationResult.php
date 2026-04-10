<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Generation;

final readonly class PromptEvaluationResult
{
    /**
     * @param list<int> $tokenIds
     */
    public function __construct(
        public array $tokenIds,
        public int $tokenCount,
        public int $durationUs,
    ) {}

    public function tokensPerSecond(): float
    {
        if ($this->durationUs <= 0 || $this->tokenCount === 0) {
            return 0.0;
        }

        return $this->tokenCount / ($this->durationUs / 1_000_000);
    }
}
