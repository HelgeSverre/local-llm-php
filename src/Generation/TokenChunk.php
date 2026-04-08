<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Generation;

final readonly class TokenChunk
{
    public function __construct(
        public int $tokenId,
        public string $text,
        public int $index,
        public int $elapsedUs,
    ) {
    }
}
