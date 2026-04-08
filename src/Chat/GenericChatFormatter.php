<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Chat;

use HelgeSverre\LocalLlm\Backend\ModelInterface;

final class GenericChatFormatter implements ChatFormatterInterface
{
    public function format(array $messages, ModelInterface $model, ChatOptions $options): string
    {
        $lines = [];

        foreach ($messages as $message) {
            $lines[] = strtoupper($message->role) . ':';
            $lines[] = $message->content;
            $lines[] = '';
        }

        if ($options->addAssistantTurn) {
            $lines[] = 'ASSISTANT:';
        } elseif ($lines !== [] && end($lines) === '') {
            array_pop($lines);
        }

        return implode("\n", $lines);
    }
}
