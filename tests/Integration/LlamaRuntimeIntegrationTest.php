<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Integration;

use HelgeSverre\LocalLlm\Backend\BackendException;
use HelgeSverre\LocalLlm\Backend\ChatTemplateAwareModelInterface;
use HelgeSverre\LocalLlm\Backend\SessionOptions;
use HelgeSverre\LocalLlm\Chat\ChatMessage;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\Runtime\LlamaRuntime;
use PHPUnit\Framework\TestCase;

final class LlamaRuntimeIntegrationTest extends TestCase
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

    public function testChatSessionUsesTheNativeTemplateAndStreamsTheSameTextAsGenerate(): void
    {
        $runtime = $this->runtime();

        try {
            $model = $runtime->model();
            if (!$model instanceof ChatTemplateAwareModelInterface || $model->defaultChatTemplate() === null) {
                self::markTestSkipped('Model does not expose a native chat template.');
            }

            $chat = $runtime->newChatSession();
            $messages = [
                ChatMessage::system('You are terse.'),
                ChatMessage::user('Say hello in three words.'),
            ];

            $formattedByChat = $chat->format($messages);
            self::assertStringContainsString('You are terse.', $formattedByChat);
            self::assertStringContainsString('Say hello in three words.', $formattedByChat);
            self::assertSame(0, preg_match('/[\x00-\x08\x0B\x0C\x0E-\x1F]/', $formattedByChat));

            $streamed = '';
            foreach ($chat->stream($messages, new GenerationConfig(maxTokens: 8, temperature: 0.0)) as $chunk) {
                $streamed .= $chunk->text;
            }

            $chat->reset();
            $generated = $chat->generate($messages, new GenerationConfig(maxTokens: 8, temperature: 0.0));

            self::assertNotSame('', $streamed);
            self::assertSame($generated->text, $streamed);
            self::assertSame($generated->text, implode('', array_map(
                static fn($chunk): string => $chunk->text,
                $generated->chunks,
            )));
        } finally {
            $runtime->close();
        }
    }

    public function testChatSessionResetProducesStableDeterministicOutput(): void
    {
        $runtime = $this->runtime();

        try {
            $chat = $runtime->newChatSession();
            $messages = [ChatMessage::user('Answer with one short word.')];
            $config = new GenerationConfig(maxTokens: 4, temperature: 0.0);

            $first = $chat->generate($messages, $config);
            $chat->reset();
            $second = $chat->generate($messages, $config);

            self::assertSame($first->text, $second->text);
        } finally {
            $runtime->close();
        }
    }

    public function testLongPromptChunkingAlsoWorksThroughChatSessions(): void
    {
        $runtime = $this->runtime(new SessionOptions(contextTokens: 4096, batchSize: 64, microBatchSize: 64));

        try {
            $chat = $runtime->newChatSession();
            $longUserPrompt = trim(str_repeat(
                'Measure prompt throughput, decode throughput, and context reuse carefully. ',
                48,
            ));

            $result = $chat->generate(
                [ChatMessage::user($longUserPrompt)],
                new GenerationConfig(maxTokens: 8, temperature: 0.0),
            );

            self::assertNotNull($result->promptEvaluation);
            self::assertGreaterThan(64, $result->promptEvaluation->tokenCount);
            self::assertGreaterThan(0, $result->profile->generatedTokens);
        } finally {
            $runtime->close();
        }
    }

    public function testGeneratingWithoutPromptOrPriorStateFailsCleanly(): void
    {
        $runtime = $this->runtime();

        try {
            $session = $runtime->newSession();

            $this->expectException(BackendException::class);
            $this->expectExceptionMessage('Generation requires a prompt or a previously evaluated session state.');

            $session->generate(new GenerationConfig(maxTokens: 4, temperature: 0.0));
        } finally {
            $runtime->close();
        }
    }

    public function testInvalidModelPathFailsCleanlyWithAValidNativeLibrary(): void
    {
        $this->expectException(BackendException::class);

        LlamaRuntime::fromModelPath(
            modelPath: '/tmp/this-model-does-not-exist.gguf',
            libraryPath: (string) getenv('LOCAL_LLM_FFI_LLAMA_LIB'),
            gpuLayers: 0,
        );
    }

    private function runtime(?SessionOptions $sessionOptions = null): LlamaRuntime
    {
        return LlamaRuntime::fromModelPath(
            modelPath: (string) getenv('LOCAL_LLM_FFI_MODEL'),
            libraryPath: (string) getenv('LOCAL_LLM_FFI_LLAMA_LIB'),
            gpuLayers: (int) (getenv('LOCAL_LLM_FFI_GPU_LAYERS') ?: '99'),
            sessionOptions: $sessionOptions,
        );
    }
}
