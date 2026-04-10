<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\FFI;

use FFI\CData;
use HelgeSverre\LocalLlm\Backend\BackendException;
use HelgeSverre\LocalLlm\Backend\ChatTemplateAwareModelInterface;
use HelgeSverre\LocalLlm\Backend\ModelInterface;
use HelgeSverre\LocalLlm\Backend\ModelMetadata;
use HelgeSverre\LocalLlm\Backend\ModelOptions;
use HelgeSverre\LocalLlm\Backend\SessionInterface;
use HelgeSverre\LocalLlm\Backend\SessionOptions;
use HelgeSverre\LocalLlm\Chat\ChatMessage;
use HelgeSverre\LocalLlm\Tokenizer\TokenizationResult;

final class LlamaModel implements ModelInterface, ChatTemplateAwareModelInterface
{
    private CData $model;
    private CData $vocab;
    private ModelMetadata $metadata;
    private bool $closed = false;
    private ?bool $nativeChatTemplateHealthy = null;

    public function __construct(
        private readonly LlamaLibrary $library,
        private readonly ModelOptions $options,
    ) {
        $ffi = $this->library->ffi();
        $params = $ffi->llama_model_default_params();
        $params->n_gpu_layers = $options->gpuLayers;
        $params->use_mmap = $options->useMmap;
        $params->use_direct_io = $options->useDirectIo;
        $params->use_mlock = $options->useMlock;
        $params->check_tensors = $options->checkTensors;
        $params->vocab_only = $options->vocabOnly;

        $model = $ffi->llama_model_load_from_file($options->modelPath, $params);
        if ($model === null || $this->library->isNull($model)) {
            throw new BackendException(sprintf('Failed to load GGUF model "%s".', $options->modelPath));
        }
        $this->model = $model;

        $this->vocab = $ffi->llama_model_get_vocab($this->model);
        $this->metadata = $this->readMetadata();
        $this->library->logger()->info('Loaded llama.cpp model.', [
            'model_path' => $options->modelPath,
            'gpu_layers' => $options->gpuLayers,
            'description' => $this->metadata->description,
            'architecture' => $this->metadata->architecture,
            'chat_template_available' => $this->metadata->chatTemplate !== null,
            'training_context_size' => $this->metadata->trainingContextSize,
            'model_size_bytes' => $this->metadata->modelSizeBytes,
        ]);
    }

    public function __destruct()
    {
        $this->close();
    }

    public function backendName(): string
    {
        $this->assertOpen();

        return 'llama.cpp';
    }

    public function metadata(): ModelMetadata
    {
        $this->assertOpen();

        return $this->metadata;
    }

    public function tokenize(string $text, bool $addSpecial = true, bool $parseSpecial = true): TokenizationResult
    {
        $this->assertOpen();

        $ffi = $this->library->ffi();
        $required = $ffi->llama_tokenize($this->vocab, $text, strlen($text), null, 0, $addSpecial, $parseSpecial);

        if ($required === \PHP_INT_MIN) {
            throw new BackendException('Tokenization overflowed int32_t.');
        }

        if ($required < 0) {
            $required = -$required;
        }

        $buffer = $this->library->new(sprintf('llama_token[%d]', max($required, 1)));
        $written = $ffi->llama_tokenize($this->vocab, $text, strlen($text), $buffer, $required, $addSpecial, $parseSpecial);

        if ($written < 0) {
            throw new BackendException('Tokenization failed on the second pass.');
        }

        $tokens = [];
        for ($index = 0; $index < $written; $index++) {
            $tokens[] = (int) $buffer[$index];
        }

        return new TokenizationResult($tokens);
    }

    public function detokenize(array $tokens, bool $removeSpecial = false, bool $unparseSpecial = true): string
    {
        $this->assertOpen();

        if ($tokens === []) {
            return '';
        }

        $ffi = $this->library->ffi();
        $tokenBuffer = $this->library->new(sprintf('llama_token[%d]', count($tokens)));
        foreach ($tokens as $index => $tokenId) {
            $tokenBuffer[$index] = $tokenId;
        }

        $capacity = max(64, count($tokens) * 8);
        while (true) {
            $textBuffer = $this->library->new(sprintf('char[%d]', $capacity));
            $written = $ffi->llama_detokenize(
                $this->vocab,
                $tokenBuffer,
                count($tokens),
                $textBuffer,
                $capacity,
                $removeSpecial,
                $unparseSpecial,
            );

            if ($written >= 0) {
                return $this->library->string($textBuffer, $written);
            }

            $capacity = -$written + 1;
        }
    }

    public function createSession(?SessionOptions $options = null): SessionInterface
    {
        $this->assertOpen();

        return new LlamaSession($this->library, $this, $options ?? new SessionOptions());
    }

    public function defaultChatTemplate(): ?string
    {
        $this->assertOpen();

        return $this->metadata->chatTemplate;
    }

    public function formatChatMessages(array $messages, ?string $template = null, bool $addAssistantTurn = true): string
    {
        $this->assertOpen();

        $template ??= $this->defaultChatTemplate();
        if ($template === null || $template === '') {
            throw new BackendException('This model does not expose a llama.cpp chat template.');
        }

        $usingDefaultTemplate = $template === $this->metadata->chatTemplate;
        if ($usingDefaultTemplate && $this->nativeChatTemplateHealthy === false) {
            throw new BackendException('The native llama.cpp chat template has already been marked unusable for this model.');
        }

        $ffi = $this->library->ffi();
        $count = count($messages);
        $chat = $this->library->new(sprintf('llama_chat_message[%d]', max($count, 1)));
        $roleBuffers = [];
        $contentBuffers = [];
        $rolePointers = [];
        $contentPointers = [];
        $estimatedLength = strlen($template) + 64;

        foreach ($messages as $index => $message) {
            if (!$message instanceof ChatMessage) {
                throw new \InvalidArgumentException('Chat messages must be instances of ChatMessage.');
            }

            $estimatedLength += (2 * strlen($message->content)) + strlen($message->role) + 16;

            $roleBuffers[$index] = $this->toCString($message->role);
            $contentBuffers[$index] = $this->toCString($message->content);
            $rolePointers[$index] = $this->library->cast('char *', $roleBuffers[$index]);
            $contentPointers[$index] = $this->library->cast('char *', $contentBuffers[$index]);
            $chat[$index]->role = $rolePointers[$index];
            $chat[$index]->content = $contentPointers[$index];
        }

        $capacity = max($estimatedLength, 256);
        while (true) {
            $buffer = $this->library->new(sprintf('char[%d]', $capacity));
            $written = (int) $ffi->llama_chat_apply_template(
                $template,
                $chat,
                $count,
                $addAssistantTurn,
                $buffer,
                $capacity,
            );

            if ($written >= 0 && $written < $capacity) {
                $formatted = $this->library->string($buffer, $written);
                $this->assertHealthyFormattedChat($messages, $formatted, $usingDefaultTemplate);

                return $formatted;
            }

            $capacity = max($capacity * 2, $written + 1);
        }
    }

    public function close(): void
    {
        if ($this->closed) {
            return;
        }

        $this->library->ffi()->llama_model_free($this->model);
        $this->closed = true;
    }

    public function modelHandle(): CData
    {
        return $this->model;
    }

    public function vocabHandle(): CData
    {
        return $this->vocab;
    }

    public function multimodalProjectorPath(): ?string
    {
        return $this->options->multimodalProjectorPath;
    }

    public function multimodalProjectorUseGpu(): bool
    {
        return $this->options->multimodalProjectorUseGpu;
    }

    private function assertOpen(): void
    {
        if ($this->closed) {
            throw new BackendException('Model has already been closed.');
        }
    }

    private function readMetadata(): ModelMetadata
    {
        $ffi = $this->library->ffi();
        $buffer = $this->library->new('char[512]');
        $descriptionLength = $ffi->llama_model_desc($this->model, $buffer, 512);
        $description = $descriptionLength > 0 ? $this->library->string($buffer, $descriptionLength) : basename($this->options->modelPath);

        return new ModelMetadata(
            description: $description,
            architecture: $this->readMetaString('general.architecture'),
            chatTemplate: $this->readChatTemplate(),
            vocabSize: (int) $ffi->llama_vocab_n_tokens($this->vocab),
            trainingContextSize: (int) $ffi->llama_model_n_ctx_train($this->model),
            embeddingSize: (int) $ffi->llama_model_n_embd($this->model),
            layerCount: (int) $ffi->llama_model_n_layer($this->model),
            headCount: (int) $ffi->llama_model_n_head($this->model),
            parameterCount: (int) $ffi->llama_model_n_params($this->model),
            modelSizeBytes: (int) $ffi->llama_model_size($this->model),
        );
    }

    private function readMetaString(string $key): ?string
    {
        $ffi = $this->library->ffi();
        $buffer = $this->library->new('char[512]');
        $written = (int) $ffi->llama_model_meta_val_str($this->model, $key, $buffer, 512);

        if ($written >= 0 && $written < 512) {
            return $this->library->string($buffer, $written);
        }

        if ($written <= 0) {
            return null;
        }

        $retry = $this->library->new(sprintf('char[%d]', $written + 1));
        $written = (int) $ffi->llama_model_meta_val_str($this->model, $key, $retry, $written + 1);

        return $written > 0 ? $this->library->string($retry, $written) : null;
    }

    private function readChatTemplate(): ?string
    {
        $ffi = $this->library->ffi();
        $pointer = $ffi->llama_model_chat_template($this->model, null);
        if (is_string($pointer)) {
            return $pointer !== '' ? $pointer : null;
        }

        if ($pointer === null || $this->library->isNull($pointer)) {
            return null;
        }

        $template = $this->library->string($pointer);

        return $template !== '' ? $template : null;
    }

    /**
     * @param list<ChatMessage> $messages
     */
    private function assertHealthyFormattedChat(array $messages, string $formatted, bool $cacheResult): void
    {
        if (preg_match('//u', $formatted) !== 1 || preg_match('/[\x00-\x08\x0B\x0C\x0E-\x1F]/', $formatted) === 1) {
            if ($cacheResult) {
                $this->nativeChatTemplateHealthy = false;
            }

            throw new BackendException('The native llama.cpp chat template produced invalid output for this model.');
        }

        foreach ($messages as $message) {
            $expectedContent = $message->content;

            if ($expectedContent !== '' && !str_contains($formatted, $expectedContent)) {
                if ($cacheResult) {
                    $this->nativeChatTemplateHealthy = false;
                }

                throw new BackendException('The native llama.cpp chat template dropped message content unexpectedly.');
            }
        }

        if ($cacheResult) {
            $this->nativeChatTemplateHealthy = true;
        }
    }

    private function toCString(string $value): CData
    {
        $buffer = $this->library->new(sprintf('char[%d]', strlen($value) + 1));
        if ($value !== '') {
            $this->library->memcpy($buffer, $value, strlen($value));
        }
        $buffer[strlen($value)] = "\0";

        return $buffer;
    }
}
