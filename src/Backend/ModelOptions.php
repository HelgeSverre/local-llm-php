<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Backend;

final readonly class ModelOptions
{
    /**
     * @param array<string, scalar|array|null> $backendOptions
     */
    public function __construct(
        public string $modelPath,
        public int $gpuLayers = 0,
        public bool $useMmap = true,
        public bool $useDirectIo = false,
        public bool $useMlock = false,
        public bool $checkTensors = false,
        public bool $vocabOnly = false,
        public array $backendOptions = [],
    ) {
        if (trim($modelPath) === '') {
            throw new \InvalidArgumentException('Model path must not be empty.');
        }

        if ($gpuLayers < -1) {
            throw new \InvalidArgumentException('GPU layers must be -1 or greater.');
        }
    }
}
