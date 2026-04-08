<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Support;

use HelgeSverre\LocalLlm\Backend\SessionOptions;

final class SupportedModelCatalog
{
    /**
     * @return list<SupportedModelProfile>
     */
    public static function all(): array
    {
        return array_values(self::map());
    }

    public static function get(string $id): SupportedModelProfile
    {
        $profile = self::map()[$id] ?? null;
        if ($profile === null) {
            throw new \InvalidArgumentException(sprintf('Unknown supported model profile "%s".', $id));
        }

        return $profile;
    }

    /**
     * @return array<string, SupportedModelProfile>
     */
    private static function map(): array
    {
        static $profiles = null;
        if ($profiles !== null) {
            return $profiles;
        }

        $profiles = [
            'granite3.3-2b-ollama' => new SupportedModelProfile(
                id: 'granite3.3-2b-ollama',
                label: 'IBM Granite 3.3 2B via Ollama',
                family: 'Granite',
                source: 'ollama',
                sourceReference: 'granite3.3:2b',
                artifact: 'Local Ollama-managed GGUF blob resolved from `ollama show --modelfile`.',
                quantization: 'Packaged Ollama quantization',
                minimumUnifiedMemoryGb: 16,
                ollamaManaged: true,
                ollamaModelName: 'granite3.3:2b',
                notes: 'Primary benchmark profile used for package vs Ollama comparisons.',
                presets: [
                    '16gb' => new AppleSiliconPreset(
                        tier: AppleSiliconTier::GB16,
                        sessionOptions: new SessionOptions(contextTokens: 4096, batchSize: 512, microBatchSize: 512),
                        gpuLayers: 99,
                        notes: 'Conservative default for 16 GB unified memory.',
                    ),
                    '32gb' => new AppleSiliconPreset(
                        tier: AppleSiliconTier::GB32,
                        sessionOptions: new SessionOptions(contextTokens: 8192, batchSize: 1024, microBatchSize: 512),
                        gpuLayers: 99,
                        notes: 'Reasonable default when unified memory allows a larger context window.',
                    ),
                    '64gb' => new AppleSiliconPreset(
                        tier: AppleSiliconTier::GB64,
                        sessionOptions: new SessionOptions(contextTokens: 16384, batchSize: 1024, microBatchSize: 512),
                        gpuLayers: 99,
                        notes: 'Useful for larger replay and evaluation workloads.',
                    ),
                ],
            ),
            'qwen2.5-3b-instruct-q5-k-m' => new SupportedModelProfile(
                id: 'qwen2.5-3b-instruct-q5-k-m',
                label: 'Qwen 2.5 3B Instruct Q5_K_M',
                family: 'Qwen 2.5',
                source: 'huggingface',
                sourceReference: 'Qwen/Qwen2.5-3B-Instruct-GGUF',
                artifact: 'qwen2.5-3b-instruct-q5_k_m.gguf',
                quantization: 'Q5_K_M',
                minimumUnifiedMemoryGb: 16,
                huggingFaceRepo: 'Qwen/Qwen2.5-3B-Instruct-GGUF',
                artifactPattern: 'qwen2.5-3b-instruct-q5_k_m.gguf',
                notes: 'Good second-family profile for validating tokenizer and chat-template behavior.',
                presets: [
                    '16gb' => new AppleSiliconPreset(
                        tier: AppleSiliconTier::GB16,
                        sessionOptions: new SessionOptions(contextTokens: 4096, batchSize: 512, microBatchSize: 512),
                    ),
                    '32gb' => new AppleSiliconPreset(
                        tier: AppleSiliconTier::GB32,
                        sessionOptions: new SessionOptions(contextTokens: 8192, batchSize: 1024, microBatchSize: 512),
                    ),
                    '64gb' => new AppleSiliconPreset(
                        tier: AppleSiliconTier::GB64,
                        sessionOptions: new SessionOptions(contextTokens: 16384, batchSize: 1024, microBatchSize: 512),
                    ),
                ],
            ),
            'qwen2.5-coder-14b-instruct-q5-k-m' => new SupportedModelProfile(
                id: 'qwen2.5-coder-14b-instruct-q5-k-m',
                label: 'Qwen 2.5 Coder 14B Instruct Q5_K_M',
                family: 'Qwen 2.5 Coder',
                source: 'huggingface',
                sourceReference: 'Qwen/Qwen2.5-Coder-14B-Instruct-GGUF',
                artifact: 'qwen2.5-coder-14b-instruct-q5_k_m-00001-of-00002.gguf + ...00002-of-00002.gguf',
                quantization: 'Q5_K_M',
                minimumUnifiedMemoryGb: 64,
                huggingFaceRepo: 'Qwen/Qwen2.5-Coder-14B-Instruct-GGUF',
                artifactPattern: 'qwen2.5-coder-14b-instruct-q5_k_m*.gguf',
                notes: 'Split GGUF coder profile for larger Apple Silicon systems.',
                presets: [
                    '64gb' => new AppleSiliconPreset(
                        tier: AppleSiliconTier::GB64,
                        sessionOptions: new SessionOptions(contextTokens: 8192, batchSize: 1024, microBatchSize: 512),
                    ),
                ],
            ),
            'granite-4.0-1b-q4-0' => new SupportedModelProfile(
                id: 'granite-4.0-1b-q4-0',
                label: 'IBM Granite 4.0 1B Q4_0',
                family: 'Granite',
                source: 'huggingface',
                sourceReference: 'ibm-granite/granite-4.0-1b-GGUF',
                artifact: 'granite-4.0-1b-Q4_0.gguf',
                quantization: 'Q4_0',
                minimumUnifiedMemoryGb: 16,
                huggingFaceRepo: 'ibm-granite/granite-4.0-1b-GGUF',
                artifactPattern: 'granite-4.0-1b-Q4_0.gguf',
                notes: 'Very small Granite profile for smoke tests and low-memory workflows.',
                presets: [
                    '16gb' => new AppleSiliconPreset(
                        tier: AppleSiliconTier::GB16,
                        sessionOptions: new SessionOptions(contextTokens: 4096, batchSize: 512, microBatchSize: 256),
                    ),
                    '32gb' => new AppleSiliconPreset(
                        tier: AppleSiliconTier::GB32,
                        sessionOptions: new SessionOptions(contextTokens: 8192, batchSize: 512, microBatchSize: 256),
                    ),
                    '64gb' => new AppleSiliconPreset(
                        tier: AppleSiliconTier::GB64,
                        sessionOptions: new SessionOptions(contextTokens: 16384, batchSize: 1024, microBatchSize: 512),
                    ),
                ],
            ),
        ];

        return $profiles;
    }
}
