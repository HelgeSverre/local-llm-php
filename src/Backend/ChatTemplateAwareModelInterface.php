<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Backend;

interface ChatTemplateAwareModelInterface extends ModelInterface
{
    public function defaultChatTemplate(): ?string;

    /**
     * @param list<\HelgeSverre\LocalLlm\Chat\ChatMessage> $messages
     */
    public function formatChatMessages(array $messages, ?string $template = null, bool $addAssistantTurn = true): string;
}
