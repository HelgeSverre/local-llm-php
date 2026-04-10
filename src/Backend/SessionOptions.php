<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Backend;

final readonly class SessionOptions
{
    /**
     * @param array<string, mixed> $backendOptions
     */
    public function __construct(
        public int $contextTokens = 4096,
        public int $batchSize = 512,
        public int $microBatchSize = 512,
        public int $maxSequences = 1,
        public int $threads = 0,
        public int $batchThreads = 0,
        public bool $flashAttention = true,
        public bool $offloadKqv = true,
        public bool $embeddings = false,
        public array $backendOptions = [],
    ) {
        if ($contextTokens <= 0) {
            throw new \InvalidArgumentException('Context tokens must be greater than zero.');
        }

        if ($batchSize <= 0) {
            throw new \InvalidArgumentException('Batch size must be greater than zero.');
        }

        if ($microBatchSize <= 0) {
            throw new \InvalidArgumentException('Micro batch size must be greater than zero.');
        }

        if ($batchSize > $contextTokens) {
            throw new \InvalidArgumentException('Batch size must not exceed context tokens.');
        }

        if ($microBatchSize > $batchSize) {
            throw new \InvalidArgumentException('Micro batch size must not exceed batch size.');
        }

        if ($maxSequences <= 0) {
            throw new \InvalidArgumentException('Max sequences must be greater than zero.');
        }

        if ($threads < 0 || $batchThreads < 0) {
            throw new \InvalidArgumentException('Thread counts must be zero or greater.');
        }
    }
}
