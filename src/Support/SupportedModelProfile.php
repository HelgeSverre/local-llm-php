<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Support;

use HelgeSverre\LocalLlm\Backend\ModelOptions;
use HelgeSverre\LocalLlm\Backend\SessionOptions;

final readonly class SupportedModelProfile
{
    /**
     * @param array<string, AppleSiliconPreset> $presets
     */
    public function __construct(
        public string $id,
        public string $label,
        public string $family,
        public string $source,
        public string $sourceReference,
        public string $artifact,
        public string $quantization,
        public int $minimumUnifiedMemoryGb,
        public bool $ollamaManaged = false,
        public ?string $ollamaModelName = null,
        public ?string $huggingFaceRepo = null,
        public ?string $artifactPattern = null,
        public string $notes = '',
        public array $presets = [],
    ) {
        if ($this->presets === []) {
            throw new \InvalidArgumentException('Supported model profiles must define at least one Apple Silicon preset.');
        }

        foreach ($this->presets as $key => $preset) {
            if (!$preset instanceof AppleSiliconPreset) {
                throw new \InvalidArgumentException('Presets must contain AppleSiliconPreset values.');
            }

            if ($key !== $preset->tier->value) {
                throw new \InvalidArgumentException(sprintf(
                    'Preset key "%s" does not match tier "%s".',
                    $key,
                    $preset->tier->value,
                ));
            }
        }
    }

    public function supportsTier(AppleSiliconTier $tier): bool
    {
        return isset($this->presets[$tier->value]);
    }

    /**
     * @return list<AppleSiliconTier>
     */
    public function supportedTiers(): array
    {
        return array_map(
            static fn(AppleSiliconPreset $preset): AppleSiliconTier => $preset->tier,
            array_values($this->presets),
        );
    }

    public function preset(AppleSiliconTier $tier): AppleSiliconPreset
    {
        if (!$this->supportsTier($tier)) {
            throw new \InvalidArgumentException(sprintf(
                'Model profile "%s" is not qualified for Apple Silicon tier "%s".',
                $this->id,
                $tier->value,
            ));
        }

        return $this->presets[$tier->value];
    }

    public function recommendedSessionOptions(AppleSiliconTier $tier): SessionOptions
    {
        return $this->preset($tier)->sessionOptions;
    }

    public function recommendedGpuLayers(AppleSiliconTier $tier): int
    {
        return $this->preset($tier)->gpuLayers;
    }

    public function modelOptions(string $modelPath, AppleSiliconTier $tier): ModelOptions
    {
        return new ModelOptions(
            modelPath: $modelPath,
            gpuLayers: $this->recommendedGpuLayers($tier),
        );
    }
}
