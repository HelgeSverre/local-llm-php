<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Integration;

use HelgeSverre\LocalLlm\Backend\BackendException;
use HelgeSverre\LocalLlm\Backend\ChatTemplateAwareModelInterface;
use HelgeSverre\LocalLlm\Backend\ModelOptions;
use HelgeSverre\LocalLlm\Backend\SessionOptions;
use HelgeSverre\LocalLlm\Chat\ChatMessage;
use HelgeSverre\LocalLlm\FFI\LlamaBackend;
use HelgeSverre\LocalLlm\FFI\LlamaBackendConfig;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\Runtime\LlamaRuntime;
use PHPUnit\Framework\TestCase;

final class LlamaBackendIntegrationTest extends TestCase
{
    protected function setUp(): void
    {
        if (getenv('LOCAL_LLM_FFI_RUN_INTEGRATION') !== '1') {
            self::markTestSkipped('Set LOCAL_LLM_FFI_RUN_INTEGRATION=1 to enable llama.cpp integration tests.');
        }

        if (!is_file((string) getenv('LOCAL_LLM_FFI_LLAMA_LIB')) || !is_file((string) getenv('LOCAL_LLM_FFI_MODEL'))) {
            self::markTestSkipped('LOCAL_LLM_FFI_LLAMA_LIB and LOCAL_LLM_FFI_MODEL must point to existing files.');
        }
    }

    public function testItCanTokenizeGenerateAndReuseSession(): void
    {
        $backend = new LlamaBackend(new LlamaBackendConfig((string) getenv('LOCAL_LLM_FFI_LLAMA_LIB')));
        $model = $backend->loadModel(new ModelOptions((string) getenv('LOCAL_LLM_FFI_MODEL'), gpuLayers: 99));
        $session = $model->createSession(new SessionOptions(contextTokens: 2048, batchSize: 256, microBatchSize: 256));

        $tokens = $model->tokenize('Hello from PHP FFI.');
        self::assertGreaterThan(0, $tokens->count());

        $prompt = $session->evaluate('Write one short word:');
        self::assertGreaterThan(0, $prompt->tokenCount);

        $first = $session->generate(new GenerationConfig(maxTokens: 4, temperature: 0.0));

        $session->reset();
        $session->evaluate('Write one short word:');
        $second = $session->generate(new GenerationConfig(maxTokens: 4, temperature: 0.0));

        self::assertSame($first->text, $second->text);

        $session->close();
        $model->close();
    }

    public function testItCanStreamAndReconstructTextDeterministically(): void
    {
        $backend = new LlamaBackend(new LlamaBackendConfig((string) getenv('LOCAL_LLM_FFI_LLAMA_LIB')));
        $model = $backend->loadModel(new ModelOptions((string) getenv('LOCAL_LLM_FFI_MODEL'), gpuLayers: 99));
        $session = $model->createSession(new SessionOptions(contextTokens: 2048, batchSize: 256, microBatchSize: 256));

        $prompt = 'Say hello in two or three words.';
        $expected = $session->generate(new GenerationConfig(prompt: $prompt, maxTokens: 8, temperature: 0.0));

        $session->reset();
        $reconstructed = '';
        foreach ($session->stream(new GenerationConfig(prompt: $prompt, maxTokens: 8, temperature: 0.0)) as $chunk) {
            $reconstructed .= $chunk->text;
        }

        self::assertSame($expected->text, $reconstructed);

        $session->close();
        $model->close();
    }

    public function testItCanUseTheModelAwareChatTemplateWhenAvailable(): void
    {
        $runtime = $this->runtime();
        $model = $runtime->model();

        if (!$model instanceof ChatTemplateAwareModelInterface || $model->defaultChatTemplate() === null) {
            self::markTestSkipped('The configured integration model does not expose a native chat template.');
        }

        $chat = $runtime->newChatSession();
        $messages = [
            ChatMessage::system('You are terse.'),
            ChatMessage::user('Say hello in three words.'),
        ];

        $prompt = $chat->format($messages);
        self::assertStringContainsString('Say hello in three words.', $prompt);

        $result = $chat->generate($messages, new GenerationConfig(maxTokens: 8, temperature: 0.0));
        self::assertGreaterThan(0, $result->profile->generatedTokens);

        $chat->close();
        $runtime->close();
    }

    public function testChatStreamingReconstructsTheSameTextAsGenerate(): void
    {
        $runtime = $this->runtime();
        $model = $runtime->model();

        if (!$model instanceof ChatTemplateAwareModelInterface || $model->defaultChatTemplate() === null) {
            self::markTestSkipped('The configured integration model does not expose a native chat template.');
        }

        $messages = [
            ChatMessage::system('You are terse.'),
            ChatMessage::user('Say hello in three words.'),
        ];

        try {
            $generateChat = $runtime->newChatSession();
            $generated = $generateChat->generate($messages, new GenerationConfig(maxTokens: 8, temperature: 0.0));
            $generateChat->close();

            $streamChat = $runtime->newChatSession();
            $reconstructed = '';
            foreach ($streamChat->stream($messages, new GenerationConfig(maxTokens: 8, temperature: 0.0)) as $chunk) {
                $reconstructed .= $chunk->text;
            }
            $streamChat->close();

            self::assertSame($generated->text, $reconstructed);
        } finally {
            $runtime->close();
        }
    }

    public function testItCanEvaluatePromptsLongerThanBatchSize(): void
    {
        $backend = new LlamaBackend(new LlamaBackendConfig((string) getenv('LOCAL_LLM_FFI_LLAMA_LIB')));
        $model = $backend->loadModel(new ModelOptions((string) getenv('LOCAL_LLM_FFI_MODEL'), gpuLayers: 99));
        $session = $model->createSession(new SessionOptions(contextTokens: 4096, batchSize: 64, microBatchSize: 64));

        $prompt = trim(str_repeat(
            'Measure prompt throughput, decode throughput, and context reuse carefully. ',
            48,
        ));

        $evaluation = $session->evaluate($prompt);

        self::assertGreaterThan(64, $evaluation->tokenCount);

        $result = $session->generate(new GenerationConfig(maxTokens: 8, temperature: 0.0));

        self::assertGreaterThan(0, $result->profile->generatedTokens);

        $session->close();
        $model->close();
    }

    public function testItFailsCleanlyForAnInvalidModelPath(): void
    {
        $backend = new LlamaBackend(new LlamaBackendConfig((string) getenv('LOCAL_LLM_FFI_LLAMA_LIB')));

        $this->expectException(BackendException::class);
        $this->expectExceptionMessage('Failed to load GGUF model');
        $backend->loadModel(new ModelOptions('/path/that/does/not/exist.gguf', gpuLayers: 99));
    }

    public function testItFailsWhenGeneratingWithoutPromptOrRestoredState(): void
    {
        $backend = new LlamaBackend(new LlamaBackendConfig((string) getenv('LOCAL_LLM_FFI_LLAMA_LIB')));
        $model = $backend->loadModel(new ModelOptions((string) getenv('LOCAL_LLM_FFI_MODEL'), gpuLayers: 99));
        $session = $model->createSession(new SessionOptions(contextTokens: 2048, batchSize: 256, microBatchSize: 256));

        try {
            $this->expectException(BackendException::class);
            $this->expectExceptionMessage('Generation requires a prompt or a previously evaluated session state.');
            $session->generate(new GenerationConfig(maxTokens: 1, temperature: 0.0));
        } finally {
            $session->close();
            $model->close();
        }
    }

    private function runtime(): LlamaRuntime
    {
        return LlamaRuntime::fromModelPath(
            modelPath: (string) getenv('LOCAL_LLM_FFI_MODEL'),
            libraryPath: (string) getenv('LOCAL_LLM_FFI_LLAMA_LIB'),
            gpuLayers: 99,
            sessionOptions: new SessionOptions(contextTokens: 2048, batchSize: 256, microBatchSize: 256),
        );
    }
}
