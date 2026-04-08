<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Generation;

final readonly class SessionState
{
    /**
     * @param list<int> $historyTokenIds
     */
    public function __construct(
        public string $bytes,
        public array $historyTokenIds,
    ) {
    }
}
