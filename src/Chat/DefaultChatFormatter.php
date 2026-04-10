<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Chat;

use HelgeSverre\LocalLlm\Backend\ChatTemplateAwareModelInterface;
use HelgeSverre\LocalLlm\Backend\ModelInterface;

final class DefaultChatFormatter implements ChatFormatterInterface
{
    public function __construct(
        private readonly ChatFormatterInterface $fallback = new GenericChatFormatter(),
    ) {}

    public function format(array $messages, ModelInterface $model, ChatOptions $options): string
    {
        if ($options->promptOverride !== null) {
            return $options->promptOverride;
        }

        if (
            $options->preferModelTemplate
            && $model instanceof ChatTemplateAwareModelInterface
            && ($options->templateOverride !== null || $model->defaultChatTemplate() !== null)
        ) {
            return $model->formatChatMessages(
                messages: $messages,
                template: $options->templateOverride,
                addAssistantTurn: $options->addAssistantTurn,
            );
        }

        return $this->fallback->format($messages, $model, $options);
    }
}
