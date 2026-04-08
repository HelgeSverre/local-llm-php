<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Chat;

final readonly class ChatMessage
{
    public function __construct(
        public string $role,
        public string $content,
    ) {
        if (trim($role) === '') {
            throw new \InvalidArgumentException('Chat message role must not be empty.');
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

    public static function assistant(string $content): self
    {
        return new self('assistant', $content);
    }

    public static function tool(string $content): self
    {
        return new self('tool', $content);
    }
}
