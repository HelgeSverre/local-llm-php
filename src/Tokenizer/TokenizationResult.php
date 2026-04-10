<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tokenizer;

final readonly class TokenizationResult
{
    /**
     * @param list<int> $tokens
     */
    public function __construct(
        public array $tokens,
    ) {}

    public function count(): int
    {
        return count($this->tokens);
    }
}
