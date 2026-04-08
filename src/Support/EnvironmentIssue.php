<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Support;

final readonly class EnvironmentIssue
{
    public function __construct(
        public string $severity,
        public string $message,
    ) {
    }

    public function isBlocking(): bool
    {
        return $this->severity === 'error';
    }
}
