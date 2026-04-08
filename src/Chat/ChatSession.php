<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Chat;

use Generator;
use HelgeSverre\LocalLlm\Backend\BackendException;
use HelgeSverre\LocalLlm\Backend\ChatTemplateAwareModelInterface;
use HelgeSverre\LocalLlm\Backend\ModelInterface;
use HelgeSverre\LocalLlm\Backend\SessionInterface;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\Generation\GenerationResult;
use HelgeSverre\LocalLlm\Generation\TokenChunk;
use Psr\Log\LoggerInterface;
use Psr\Log\NullLogger;

final class ChatSession
{
    public function __construct(
        private readonly ModelInterface $model,
        private readonly SessionInterface $session,
        private readonly ChatFormatterInterface $formatter = new DefaultChatFormatter(),
        private readonly LoggerInterface $logger = new NullLogger(),
    ) {
    }

    /**
     * @param list<ChatMessage> $messages
     */
    public function format(array $messages, ?ChatOptions $options = null): string
    {
        $options ??= new ChatOptions();
        $formatter = $options->formatterOverride ?? $this->formatter;
        $templateMode = 'generic';

        try {
            $prompt = $formatter->format($messages, $this->model, $options);

            if ($options->promptOverride !== null) {
                $templateMode = 'prompt_override';
            } elseif ($options->formatterOverride !== null) {
                $templateMode = 'formatter_override';
            } elseif (
                $options->preferModelTemplate
                && $this->model instanceof ChatTemplateAwareModelInterface
                && ($options->templateOverride !== null || $this->model->defaultChatTemplate() !== null)
            ) {
                $templateMode = $options->templateOverride !== null ? 'template_override' : 'model_native';
            }
        } catch (BackendException $exception) {
            if (
                $options->formatterOverride !== null
                || !$options->preferModelTemplate
                || !$this->model instanceof ChatTemplateAwareModelInterface
                || ($options->templateOverride === null && $this->model->defaultChatTemplate() === null)
            ) {
                throw $exception;
            }

            $prompt = (new GenericChatFormatter())->format($messages, $this->model, $options);
            $templateMode = 'native_fallback_generic';
            $this->logger->warning('Fell back to generic chat formatting after native template failure.', [
                'message_count' => count($messages),
                'formatter' => $formatter::class,
                'error' => $exception->getMessage(),
            ]);
        }

        $this->logger->debug('Formatted chat prompt.', [
            'message_count' => count($messages),
            'prompt_length' => strlen($prompt),
            'formatter' => $formatter::class,
            'template_mode' => $templateMode,
        ]);

        return $prompt;
    }

    /**
     * @param list<ChatMessage> $messages
     * @param callable(TokenChunk): void|null $onToken
     */
    public function generate(
        array $messages,
        ?GenerationConfig $config = null,
        ?callable $onToken = null,
        ?ChatOptions $options = null,
    ): GenerationResult {
        $prompt = $this->format($messages, $options);

        return $this->session->generate($this->withPrompt($config ?? new GenerationConfig(), $prompt), $onToken);
    }

    /**
     * @param list<ChatMessage> $messages
     * @return Generator<int, TokenChunk>
     */
    public function stream(
        array $messages,
        ?GenerationConfig $config = null,
        ?ChatOptions $options = null,
    ): Generator {
        $prompt = $this->format($messages, $options);

        return $this->session->stream($this->withPrompt($config ?? new GenerationConfig(), $prompt));
    }

    public function reset(bool $clearStateData = true): void
    {
        $this->session->reset($clearStateData);
    }

    public function lowLevelSession(): SessionInterface
    {
        return $this->session;
    }

    public function close(): void
    {
        $this->session->close();
    }

    private function withPrompt(GenerationConfig $config, string $prompt): GenerationConfig
    {
        return new GenerationConfig(
            prompt: $prompt,
            maxTokens: $config->maxTokens,
            temperature: $config->temperature,
            topK: $config->topK,
            topP: $config->topP,
            minP: $config->minP,
            seed: $config->seed,
            addSpecial: $config->addSpecial,
            parseSpecial: $config->parseSpecial,
            removeSpecialOnDetokenize: $config->removeSpecialOnDetokenize,
            unparseSpecial: $config->unparseSpecial,
            stopStrings: $config->stopStrings,
            stopTokens: $config->stopTokens,
        );
    }
}
