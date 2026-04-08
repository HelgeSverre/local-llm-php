<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Chat;

use HelgeSverre\LocalLlm\Backend\ModelInterface;

interface ChatFormatterInterface
{
    /**
     * @param list<ChatMessage> $messages
     */
    public function format(array $messages, ModelInterface $model, ChatOptions $options): string;
}
