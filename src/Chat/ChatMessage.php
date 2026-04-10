<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Chat;

use HelgeSverre\LocalLlm\Generation\MediaInput;

final readonly class ChatMessage
{
    /**
     * @param list<MediaInput> $mediaInputs
     */
    public function __construct(
        public string $role,
        public string $content,
        public array $mediaInputs = [],
    ) {
        if (trim($role) === '') {
            throw new \InvalidArgumentException('Chat message role must not be empty.');
        }

        foreach ($mediaInputs as $mediaInput) {
            if (!$mediaInput instanceof MediaInput) {
                throw new \InvalidArgumentException('Chat media inputs must be instances of ' . MediaInput::class . '.');
            }
        }
    }

    public static function system(string $content): self
    {
        return new self('system', $content);
    }

    public static function user(string $content): self
    {
        return new self('user', $content);
    }

    /**
     * @param list<MediaInput> $mediaInputs
     */
    public static function userWithMedia(string $content, array $mediaInputs): self
    {
        return new self('user', $content, $mediaInputs);
    }

    public static function assistant(string $content): self
    {
        return new self('assistant', $content);
    }

    public static function tool(string $content): self
    {
        return new self('tool', $content);
    }

    /**
     * @param list<MediaInput> $mediaInputs
     */
    public function withMedia(array $mediaInputs): self
    {
        return new self($this->role, $this->content, $mediaInputs);
    }

    public function contentForPrompt(string $mediaMarker = MediaInput::DEFAULT_MARKER): string
    {
        if ($this->mediaInputs === []) {
            return $this->content;
        }

        $content = $this->content;
        $presentMarkerCount = substr_count($content, $mediaMarker);
        $missingMarkerCount = count($this->mediaInputs) - $presentMarkerCount;

        if ($missingMarkerCount <= 0) {
            return $content;
        }

        $suffix = implode("\n", array_fill(0, $missingMarkerCount, $mediaMarker));

        if ($content === '') {
            return $suffix;
        }

        return rtrim($content) . "\n" . $suffix;
    }
}
