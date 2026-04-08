<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Generation;

final readonly class GenerationResult
{
    /**
     * @param list<int> $tokenIds
     * @param list<TokenChunk> $chunks
     */
    public function __construct(
        public string $text,
        public array $tokenIds,
        public array $chunks,
        public string $finishReason,
        public ?PromptEvaluationResult $promptEvaluation,
        public GenerationProfile $profile,
    ) {
    }
}
