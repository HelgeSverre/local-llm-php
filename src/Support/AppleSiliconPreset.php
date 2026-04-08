<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Support;

use HelgeSverre\LocalLlm\Backend\SessionOptions;

final readonly class AppleSiliconPreset
{
    public function __construct(
        public AppleSiliconTier $tier,
        public SessionOptions $sessionOptions,
        public int $gpuLayers = 99,
        public string $notes = '',
    ) {
        if ($gpuLayers < -1) {
            throw new \InvalidArgumentException('GPU layers must be -1 or greater.');
        }
    }
}
