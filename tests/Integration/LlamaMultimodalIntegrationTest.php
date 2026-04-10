<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Integration;

use HelgeSverre\LocalLlm\Backend\BackendException;
use HelgeSverre\LocalLlm\Backend\MediaAwareSessionInterface;
use HelgeSverre\LocalLlm\Backend\SessionOptions;
use HelgeSverre\LocalLlm\Chat\ChatMessage;
use HelgeSverre\LocalLlm\Chat\ChatOptions;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\Generation\MediaInput;
use HelgeSverre\LocalLlm\Runtime\LlamaRuntime;
use HelgeSverre\LocalLlm\Support\RuntimePlatform;
use PHPUnit\Framework\TestCase;

final class LlamaMultimodalIntegrationTest extends TestCase
{
    protected function setUp(): void
    {
        if (getenv('LOCAL_LLM_FFI_RUN_INTEGRATION') !== '1') {
            self::markTestSkipped('Set LOCAL_LLM_FFI_RUN_INTEGRATION=1 to enable llama.cpp integration tests.');
        }

        if (!is_file((string) getenv('LOCAL_LLM_FFI_LLAMA_LIB'))) {
            self::markTestSkipped('LOCAL_LLM_FFI_LLAMA_LIB must point to an existing shared library.');
        }

        $libraryPath = (string) getenv('LOCAL_LLM_FFI_LLAMA_LIB');
        if (
            basename($libraryPath) === RuntimePlatform::sharedLibraryBasename()
            && !is_file(dirname($libraryPath) . '/' . RuntimePlatform::multimodalSharedLibraryBasename())
        ) {
            self::markTestSkipped('Multimodal integration requires a sibling ' . RuntimePlatform::multimodalSharedLibraryBasename() . ' next to LOCAL_LLM_FFI_LLAMA_LIB, or LOCAL_LLM_FFI_LLAMA_LIB must point to that library directly.');
        }

        if (
            !is_file((string) getenv('LOCAL_LLM_FFI_MM_MODEL'))
            || !is_file((string) getenv('LOCAL_LLM_FFI_MM_MMPROJ'))
            || !is_file((string) getenv('LOCAL_LLM_FFI_MM_IMAGE'))
        ) {
            self::markTestSkipped(
                'LOCAL_LLM_FFI_MM_MODEL, LOCAL_LLM_FFI_MM_MMPROJ, and LOCAL_LLM_FFI_MM_IMAGE must point to existing files.',
            );
        }
    }

    public function testItCanGenerateFromAChatMessageWithImageInput(): void
    {
        $runtime = $this->runtime();

        try {
            $chat = $runtime->newChatSession();
            try {
                $result = $chat->generate(
                    [
                        ChatMessage::userWithMedia(
                            'Describe the attached image in a few words.',
                            [MediaInput::fromFile((string) getenv('LOCAL_LLM_FFI_MM_IMAGE'))],
                        ),
                    ],
                    new GenerationConfig(maxTokens: 12, temperature: 0.0),
                );

                self::assertNotNull($result->promptEvaluation);
                self::assertGreaterThan(0, $result->promptEvaluation->tokenCount);
                self::assertGreaterThan(0, $result->profile->generatedTokens);
                self::assertNotSame('', trim($result->text));
            } finally {
                $chat->close();
            }
        } finally {
            $runtime->close();
        }
    }

    public function testItCanEvaluateAndGenerateThroughTheLowLevelSessionApi(): void
    {
        $runtime = $this->runtime();

        try {
            $session = $runtime->newSession();
            try {
                $mediaMarker = $this->mediaMarkerForSession($session);
                $result = $session->generate(new GenerationConfig(
                    prompt: "Describe this image.\n" . $mediaMarker,
                    mediaInputs: [MediaInput::fromFile((string) getenv('LOCAL_LLM_FFI_MM_IMAGE'))],
                    maxTokens: 12,
                    temperature: 0.0,
                ));

                self::assertNotNull($result->promptEvaluation);
                self::assertGreaterThan(0, $result->promptEvaluation->tokenCount);
                self::assertGreaterThan(0, $result->profile->generatedTokens);
                self::assertNotSame('', trim($result->text));
            } finally {
                $session->close();
            }
        } finally {
            $runtime->close();
        }
    }

    public function testItCanStreamTheSameMultimodalChatTextAsGenerate(): void
    {
        $runtime = $this->runtime();
        $messages = [
            ChatMessage::userWithMedia(
                'What animal appears in this image? Answer in a few words.',
                [MediaInput::fromFile((string) getenv('LOCAL_LLM_FFI_MM_IMAGE'))],
            ),
        ];
        $config = new GenerationConfig(maxTokens: 12, temperature: 0.0);

        try {
            $generateChat = $runtime->newChatSession();
            $generated = $generateChat->generate($messages, $config);
            $generateChat->close();

            $streamChat = $runtime->newChatSession();
            $streamed = '';
            foreach ($streamChat->stream($messages, $config) as $chunk) {
                $streamed .= $chunk->text;
            }
            $streamChat->close();

            self::assertNotSame('', trim($streamed));
            self::assertSame($generated->text, $streamed);
        } finally {
            $runtime->close();
        }
    }

    public function testItFailsCleanlyWhenThePromptMarkerCountDoesNotMatchTheMediaInputs(): void
    {
        $runtime = $this->runtime();

        try {
            $session = $runtime->newSession();
            try {
                $this->expectException(BackendException::class);
                $this->expectExceptionMessage('Multimodal prompt must contain exactly 1');

                $session->generate(new GenerationConfig(
                    prompt: 'Describe this image.',
                    mediaInputs: [MediaInput::fromFile((string) getenv('LOCAL_LLM_FFI_MM_IMAGE'))],
                    maxTokens: 12,
                    temperature: 0.0,
                ));
            } finally {
                $session->close();
            }
        } finally {
            $runtime->close();
        }
    }

    public function testItRejectsPromptOverrideWhenChatMessagesAlreadyCarryMedia(): void
    {
        $runtime = $this->runtime();

        try {
            $chat = $runtime->newChatSession();
            try {
                $this->expectException(BackendException::class);
                $this->expectExceptionMessage('Chat message media inputs cannot be combined with prompt override.');

                $chat->generate(
                    [
                        ChatMessage::userWithMedia(
                            'Describe this image.',
                            [MediaInput::fromFile((string) getenv('LOCAL_LLM_FFI_MM_IMAGE'))],
                        ),
                    ],
                    new GenerationConfig(maxTokens: 12, temperature: 0.0),
                    options: new ChatOptions(promptOverride: 'RAW PROMPT'),
                );
            } finally {
                $chat->close();
            }
        } finally {
            $runtime->close();
        }
    }

    public function testItRejectsMixingChatMessageMediaAndGenerationConfigMediaInputs(): void
    {
        $runtime = $this->runtime();

        try {
            $chat = $runtime->newChatSession();
            try {
                $this->expectException(BackendException::class);
                $this->expectExceptionMessage('Attach media inputs either to chat messages or to GenerationConfig');

                $chat->generate(
                    [
                        ChatMessage::userWithMedia(
                            'Describe this image.',
                            [MediaInput::fromFile((string) getenv('LOCAL_LLM_FFI_MM_IMAGE'))],
                        ),
                    ],
                    new GenerationConfig(
                        maxTokens: 12,
                        temperature: 0.0,
                        mediaInputs: [MediaInput::fromFile((string) getenv('LOCAL_LLM_FFI_MM_IMAGE'))],
                    ),
                );
            } finally {
                $chat->close();
            }
        } finally {
            $runtime->close();
        }
    }

    public function testItCanContinueGenerationAfterAMultimodalPromptOnlyEvaluation(): void
    {
        $runtime = $this->runtime();

        try {
            $session = $runtime->newSession();
            try {
                $mediaMarker = $this->mediaMarkerForSession($session);
                $promptEvaluationOnly = new GenerationConfig(
                    prompt: "What animal appears in this image?\n" . $mediaMarker,
                    mediaInputs: [MediaInput::fromFile((string) getenv('LOCAL_LLM_FFI_MM_IMAGE'))],
                    maxTokens: 0,
                    temperature: 0.0,
                );

                $initial = $session->generate($promptEvaluationOnly);
                $continued = $session->generate(new GenerationConfig(maxTokens: 8, temperature: 0.0));

                self::assertNotNull($initial->promptEvaluation);
                self::assertGreaterThan(0, $initial->promptEvaluation->tokenCount);
                self::assertSame(0, $initial->profile->generatedTokens);
                self::assertGreaterThan(0, $continued->profile->generatedTokens);
                self::assertNotSame('', trim($continued->text));
            } finally {
                $session->close();
            }
        } finally {
            $runtime->close();
        }
    }

    public function testItFailsCleanlyWhenSnapshottingMultimodalPromptStateOnDarwin(): void
    {
        if (RuntimePlatform::osFamily() !== 'Darwin') {
            self::markTestSkipped('This guard is specific to the current Darwin/Metal multimodal runtime path.');
        }

        $runtime = $this->runtime();

        try {
            $session = $runtime->newSession();
            try {
                $mediaMarker = $this->mediaMarkerForSession($session);
                $session->generate(new GenerationConfig(
                    prompt: "What animal appears in this image?\n" . $mediaMarker,
                    mediaInputs: [MediaInput::fromFile((string) getenv('LOCAL_LLM_FFI_MM_IMAGE'))],
                    maxTokens: 0,
                    temperature: 0.0,
                ));

                $this->expectException(BackendException::class);
                $this->expectExceptionMessage('Serialized snapshot/restore is not supported after multimodal prompt evaluation');

                $session->snapshot();
            } finally {
                $session->close();
            }
        } finally {
            $runtime->close();
        }
    }

    private function runtime(): LlamaRuntime
    {
        return LlamaRuntime::fromModelPath(
            modelPath: (string) getenv('LOCAL_LLM_FFI_MM_MODEL'),
            libraryPath: (string) getenv('LOCAL_LLM_FFI_LLAMA_LIB'),
            gpuLayers: (int) (getenv('LOCAL_LLM_FFI_MM_GPU_LAYERS') ?: getenv('LOCAL_LLM_FFI_GPU_LAYERS') ?: '99'),
            sessionOptions: new SessionOptions(contextTokens: 2048, batchSize: 256, microBatchSize: 256),
            multimodalProjectorPath: (string) getenv('LOCAL_LLM_FFI_MM_MMPROJ'),
        );
    }

    private function mediaMarkerForSession(object $session): string
    {
        if ($session instanceof MediaAwareSessionInterface && $session->mediaMarker() !== null) {
            return $session->mediaMarker();
        }

        return MediaInput::DEFAULT_MARKER;
    }
}
