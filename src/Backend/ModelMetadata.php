<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Backend;

final readonly class ModelMetadata
{
    public function __construct(
        public string $description,
        public ?string $architecture,
        public ?string $chatTemplate,
        public int $vocabSize,
        public int $trainingContextSize,
        public int $embeddingSize,
        public int $layerCount,
        public int $headCount,
        public int $parameterCount,
        public int $modelSizeBytes,
    ) {}
}
