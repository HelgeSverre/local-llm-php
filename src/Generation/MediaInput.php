<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Generation;

final readonly class MediaInput
{
    public const DEFAULT_MARKER = '<__media__>';

    public function __construct(
        public string $path,
        public ?string $id = null,
    ) {
        if (trim($path) === '') {
            throw new \InvalidArgumentException('Media input path must not be empty.');
        }
    }

    public static function fromFile(string $path, ?string $id = null): self
    {
        return new self($path, $id);
    }
}
