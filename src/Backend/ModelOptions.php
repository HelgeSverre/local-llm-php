<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Backend;

final readonly class ModelOptions
{
    /**
     * @param array<string, mixed> $backendOptions
     */
    public function __construct(
        public string $modelPath,
        public int $gpuLayers = 0,
        public bool $useMmap = true,
        public bool $useDirectIo = false,
        public bool $useMlock = false,
        public bool $checkTensors = false,
        public bool $vocabOnly = false,
        public ?string $multimodalProjectorPath = null,
        public bool $multimodalProjectorUseGpu = true,
        public array $backendOptions = [],
    ) {
        if (trim($modelPath) === '') {
            throw new \InvalidArgumentException('Model path must not be empty.');
        }

        if ($gpuLayers < -1) {
            throw new \InvalidArgumentException('GPU layers must be -1 or greater.');
        }

        if ($multimodalProjectorPath !== null && trim($multimodalProjectorPath) === '') {
            throw new \InvalidArgumentException('Multimodal projector path must not be empty when provided.');
        }
    }
}
