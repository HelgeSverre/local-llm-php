<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Support;

enum AppleSiliconTier: string
{
    case GB16 = '16gb';
    case GB32 = '32gb';
    case GB64 = '64gb';

    public function unifiedMemoryGb(): int
    {
        return match ($this) {
            self::GB16 => 16,
            self::GB32 => 32,
            self::GB64 => 64,
        };
    }
}
