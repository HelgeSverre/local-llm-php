<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Chat;

use Generator;
use HelgeSverre\LocalLlm\Backend\BackendException;
use HelgeSverre\LocalLlm\Backend\ChatTemplateAwareModelInterface;
use HelgeSverre\LocalLlm\Backend\MediaAwareSessionInterface;
use HelgeSverre\LocalLlm\Backend\ModelInterface;
use HelgeSverre\LocalLlm\Backend\SessionInterface;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\Generation\GenerationResult;
use HelgeSverre\LocalLlm\Generation\MediaInput;
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
    ) {}

    /**
     * @param list<ChatMessage> $messages
     */
    public function format(array $messages, ?ChatOptions $options = null): string
    {
        return $this->formatMessages($messages, $options, $this->resolveMediaMarker());
    }

    /**
     * @param list<ChatMessage> $messages
     */
    private function formatMessages(array $messages, ?ChatOptions $options, string $mediaMarker): string
    {
        $options ??= new ChatOptions();
        $messages = $this->normalizeMessages($messages, $mediaMarker);
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
        $config ??= new GenerationConfig();
        $this->assertSupportedMediaBinding($messages, $config, $options);
        $mediaMarker = $this->resolveMediaMarker($config);
        $messages = $this->normalizeMessages($messages, $mediaMarker);
        $prompt = $this->formatMessages($messages, $options, $mediaMarker);

        return $this->session->generate($this->withPrompt($config, $prompt, $this->collectMediaInputs($messages)), $onToken);
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
        $config ??= new GenerationConfig();
        $this->assertSupportedMediaBinding($messages, $config, $options);
        $mediaMarker = $this->resolveMediaMarker($config);
        $messages = $this->normalizeMessages($messages, $mediaMarker);
        $prompt = $this->formatMessages($messages, $options, $mediaMarker);

        return $this->session->stream($this->withPrompt($config, $prompt, $this->collectMediaInputs($messages)));
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

    /**
     * @param list<MediaInput> $messageMediaInputs
     */
    private function withPrompt(GenerationConfig $config, string $prompt, array $messageMediaInputs): GenerationConfig
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
            mediaInputs: [...$messageMediaInputs, ...$config->mediaInputs],
            mediaMarker: $config->mediaMarker,
        );
    }

    /**
     * @param list<ChatMessage> $messages
     * @return list<ChatMessage>
     */
    private function normalizeMessages(array $messages, string $mediaMarker): array
    {
        return array_map(
            static fn(ChatMessage $message): ChatMessage => new ChatMessage(
                role: $message->role,
                content: $message->contentForPrompt($mediaMarker),
                mediaInputs: $message->mediaInputs,
            ),
            $messages,
        );
    }

    /**
     * @param list<ChatMessage> $messages
     * @return list<MediaInput>
     */
    private function collectMediaInputs(array $messages): array
    {
        $mediaInputs = [];

        foreach ($messages as $message) {
            array_push($mediaInputs, ...$message->mediaInputs);
        }

        return $mediaInputs;
    }

    /**
     * @param list<ChatMessage> $messages
     */
    private function assertSupportedMediaBinding(array $messages, GenerationConfig $config, ?ChatOptions $options): void
    {
        $messageMediaInputs = $this->collectMediaInputs($messages);
        if ($messageMediaInputs === []) {
            return;
        }

        if ($options?->promptOverride !== null) {
            throw new BackendException(
                'Chat message media inputs cannot be combined with prompt override. Use GenerationConfig media inputs with the overridden prompt instead.',
            );
        }

        if ($config->mediaInputs !== []) {
            throw new BackendException(
                'Attach media inputs either to chat messages or to GenerationConfig, not both in the same chat request.',
            );
        }
    }

    private function resolveMediaMarker(?GenerationConfig $config = null): string
    {
        if ($config?->mediaMarker !== null) {
            return $config->mediaMarker;
        }

        if ($this->session instanceof MediaAwareSessionInterface && $this->session->mediaMarker() !== null) {
            return $this->session->mediaMarker();
        }

        return MediaInput::DEFAULT_MARKER;
    }
}
