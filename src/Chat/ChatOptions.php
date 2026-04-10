<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Chat;

final readonly class ChatOptions
{
    public function __construct(
        public bool $preferModelTemplate = true,
        public bool $addAssistantTurn = true,
        public ?string $templateOverride = null,
        public ?string $promptOverride = null,
        public ?ChatFormatterInterface $formatterOverride = null,
    ) {}
}
