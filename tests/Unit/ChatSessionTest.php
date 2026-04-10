<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use Generator;
use HelgeSverre\LocalLlm\Backend\BackendException;
use HelgeSverre\LocalLlm\Backend\ChatTemplateAwareModelInterface;
use HelgeSverre\LocalLlm\Backend\MediaAwareSessionInterface;
use HelgeSverre\LocalLlm\Backend\ModelMetadata;
use HelgeSverre\LocalLlm\Backend\SessionInterface;
use HelgeSverre\LocalLlm\Chat\ChatMessage;
use HelgeSverre\LocalLlm\Chat\ChatOptions;
use HelgeSverre\LocalLlm\Chat\ChatSession;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\Generation\GenerationProfile;
use HelgeSverre\LocalLlm\Generation\GenerationResult;
use HelgeSverre\LocalLlm\Generation\MediaInput;
use HelgeSverre\LocalLlm\Generation\PromptEvaluationResult;
use HelgeSverre\LocalLlm\Generation\SessionState;
use HelgeSverre\LocalLlm\Tokenizer\TokenizationResult;
use PHPUnit\Framework\TestCase;
use Psr\Log\LoggerInterface;

final class ChatSessionTest extends TestCase
{
    public function testItUsesTheModelNativeTemplateByDefault(): void
    {
        $model = new FakeChatTemplateAwareModel();
        $session = new FakeSession();
        $chatSession = new ChatSession($model, $session);

        $prompt = $chatSession->format([
            ChatMessage::system('Be concise.'),
            ChatMessage::user('Say hello.'),
        ]);

        self::assertStringContainsString('NATIVE_TEMPLATE', $prompt);
    }

    public function testItFallsBackToGenericFormattingWhenRequested(): void
    {
        $model = new FakeChatTemplateAwareModel();
        $session = new FakeSession();
        $chatSession = new ChatSession($model, $session);

        $prompt = $chatSession->format(
            [ChatMessage::user('Say hello.')],
            new ChatOptions(preferModelTemplate: false),
        );

        self::assertStringContainsString("USER:\nSay hello.", $prompt);
        self::assertStringContainsString('ASSISTANT:', $prompt);
    }

    public function testPromptOverrideWinsCompletely(): void
    {
        $model = new FakeChatTemplateAwareModel();
        $session = new FakeSession();
        $chatSession = new ChatSession($model, $session);

        $prompt = $chatSession->format(
            [ChatMessage::user('ignored')],
            new ChatOptions(promptOverride: 'RAW PROMPT'),
        );

        self::assertSame('RAW PROMPT', $prompt);
    }

    public function testGenerateUsesFormattedChatPrompt(): void
    {
        $model = new FakeChatTemplateAwareModel();
        $session = new FakeSession();
        $chatSession = new ChatSession($model, $session);

        $chatSession->generate(
            [ChatMessage::user('Hello')],
            new GenerationConfig(maxTokens: 4, temperature: 0.0),
        );

        self::assertStringContainsString('NATIVE_TEMPLATE', $session->lastPrompt ?? '');
    }

    public function testTemplateOverrideAndAssistantFlagArePassedToTheModelFormatter(): void
    {
        $model = new FakeChatTemplateAwareModel();
        $session = new FakeSession();
        $chatSession = new ChatSession($model, $session);

        $prompt = $chatSession->format(
            [ChatMessage::user('Hello')],
            new ChatOptions(templateOverride: '{{ custom }}', addAssistantTurn: false),
        );

        self::assertStringContainsString('template={{ custom }}', $prompt);
        self::assertStringContainsString('add_assistant=0', $prompt);
        self::assertSame('{{ custom }}', $model->lastTemplate);
        self::assertFalse($model->lastAddAssistantTurn);
    }

    public function testItLogsTheSelectedFormattingMode(): void
    {
        $model = new FakeChatTemplateAwareModel();
        $session = new FakeSession();
        $logger = new FakeLogger();
        $chatSession = new ChatSession($model, $session, logger: $logger);

        $chatSession->format(
            [ChatMessage::user('Hello')],
            new ChatOptions(promptOverride: 'RAW PROMPT'),
        );

        self::assertCount(1, $logger->records);
        self::assertSame('Formatted chat prompt.', $logger->records[0]['message']);
        self::assertSame('prompt_override', $logger->records[0]['context']['template_mode']);
    }

    public function testItFallsBackToTheGenericFormatterWhenNativeFormattingFails(): void
    {
        $model = new BrokenChatTemplateAwareModel();
        $session = new FakeSession();
        $logger = new FakeLogger();
        $chatSession = new ChatSession($model, $session, logger: $logger);

        $prompt = $chatSession->format([ChatMessage::user('Hello')]);

        self::assertStringContainsString("USER:\nHello", $prompt);
        self::assertStringContainsString('ASSISTANT:', $prompt);
        self::assertSame('warning', $logger->records[0]['level']);
        self::assertSame('native_fallback_generic', $logger->records[1]['context']['template_mode']);
    }

    public function testGeneratePropagatesMessageMediaInputsAndInjectsMarkers(): void
    {
        $model = new FakeChatTemplateAwareModel();
        $session = new FakeMediaAwareSession('<native-media>');
        $chatSession = new ChatSession($model, $session);

        $chatSession->generate(
            [ChatMessage::userWithMedia('Describe this image.', [MediaInput::fromFile('/tmp/example.png')])],
            new GenerationConfig(maxTokens: 4, temperature: 0.0),
        );

        self::assertNotNull($session->lastConfig);
        self::assertCount(1, $session->lastConfig->mediaInputs);
        self::assertStringContainsString('<native-media>', $session->lastPrompt ?? '');
    }

    public function testPromptOverrideRejectsMessageAttachedMediaInputs(): void
    {
        $model = new FakeChatTemplateAwareModel();
        $session = new FakeSession();
        $chatSession = new ChatSession($model, $session);

        $this->expectException(BackendException::class);
        $this->expectExceptionMessage('Chat message media inputs cannot be combined with prompt override.');

        $chatSession->generate(
            [ChatMessage::userWithMedia('Describe this image.', [MediaInput::fromFile('/tmp/example.png')])],
            new GenerationConfig(maxTokens: 4, temperature: 0.0),
            options: new ChatOptions(promptOverride: 'RAW PROMPT'),
        );
    }

    public function testMixingMessageMediaInputsAndGenerationConfigMediaInputsFails(): void
    {
        $model = new FakeChatTemplateAwareModel();
        $session = new FakeSession();
        $chatSession = new ChatSession($model, $session);

        $this->expectException(BackendException::class);
        $this->expectExceptionMessage('Attach media inputs either to chat messages or to GenerationConfig');

        $chatSession->generate(
            [ChatMessage::userWithMedia('Describe this image.', [MediaInput::fromFile('/tmp/example.png')])],
            new GenerationConfig(
                maxTokens: 4,
                temperature: 0.0,
                mediaInputs: [MediaInput::fromFile('/tmp/other.png')],
            ),
        );
    }
}

class FakeChatTemplateAwareModel implements ChatTemplateAwareModelInterface
{
    public ?string $lastTemplate = null;
    public bool $lastAddAssistantTurn = true;

    public function backendName(): string
    {
        return 'fake';
    }

    public function metadata(): ModelMetadata
    {
        return new ModelMetadata(
            description: 'fake',
            architecture: 'fake',
            chatTemplate: '{{ chat }}',
            vocabSize: 0,
            trainingContextSize: 0,
            embeddingSize: 0,
            layerCount: 0,
            headCount: 0,
            parameterCount: 0,
            modelSizeBytes: 0,
        );
    }

    public function tokenize(string $text, bool $addSpecial = true, bool $parseSpecial = true): TokenizationResult
    {
        return new TokenizationResult([]);
    }

    public function detokenize(array $tokens, bool $removeSpecial = false, bool $unparseSpecial = true): string
    {
        return '';
    }

    public function createSession(?\HelgeSverre\LocalLlm\Backend\SessionOptions $options = null): SessionInterface
    {
        return new FakeSession();
    }

    public function close(): void {}

    public function defaultChatTemplate(): ?string
    {
        return '{{ chat }}';
    }

    public function formatChatMessages(array $messages, ?string $template = null, bool $addAssistantTurn = true): string
    {
        $this->lastTemplate = $template;
        $this->lastAddAssistantTurn = $addAssistantTurn;

        return 'NATIVE_TEMPLATE:template=' . ($template ?? 'default') . ':add_assistant=' . ($addAssistantTurn ? '1' : '0') . ':' . implode('|', array_map(
            static fn(ChatMessage $message): string => $message->role . '=' . $message->content,
            $messages,
        ));
    }
}

class FakeSession implements SessionInterface
{
    public ?string $lastPrompt = null;
    public ?GenerationConfig $lastConfig = null;

    public function evaluate(string|array $prompt, bool $addSpecial = true, bool $parseSpecial = true): PromptEvaluationResult
    {
        return new PromptEvaluationResult([], 0, 0);
    }

    public function generate(GenerationConfig $config, ?callable $onToken = null): GenerationResult
    {
        $this->lastPrompt = $config->prompt;
        $this->lastConfig = $config;

        return new GenerationResult(
            text: '',
            tokenIds: [],
            chunks: [],
            finishReason: 'stop',
            promptEvaluation: new PromptEvaluationResult([], 0, 0),
            profile: new GenerationProfile(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0),
        );
    }

    public function stream(GenerationConfig $config): Generator
    {
        yield from [];
    }

    public function snapshot(): SessionState
    {
        return new SessionState('', []);
    }

    public function restore(SessionState $state): void {}

    public function reset(bool $clearStateData = true): void {}

    public function close(): void {}
}

final class FakeMediaAwareSession extends FakeSession implements MediaAwareSessionInterface
{
    public function __construct(
        private readonly ?string $marker,
    ) {}

    public function mediaMarker(): ?string
    {
        return $this->marker;
    }
}

final class BrokenChatTemplateAwareModel extends FakeChatTemplateAwareModel
{
    public function formatChatMessages(array $messages, ?string $template = null, bool $addAssistantTurn = true): string
    {
        throw new \HelgeSverre\LocalLlm\Backend\BackendException('native template failed');
    }
}

final class FakeLogger implements LoggerInterface
{
    /** @var list<array{level:string,message:string,context:array<mixed>}> */
    public array $records = [];

    public function emergency(\Stringable|string $message, array $context = []): void
    {
        $this->log('emergency', $message, $context);
    }

    public function alert(\Stringable|string $message, array $context = []): void
    {
        $this->log('alert', $message, $context);
    }

    public function critical(\Stringable|string $message, array $context = []): void
    {
        $this->log('critical', $message, $context);
    }

    public function error(\Stringable|string $message, array $context = []): void
    {
        $this->log('error', $message, $context);
    }

    public function warning(\Stringable|string $message, array $context = []): void
    {
        $this->log('warning', $message, $context);
    }

    public function notice(\Stringable|string $message, array $context = []): void
    {
        $this->log('notice', $message, $context);
    }

    public function info(\Stringable|string $message, array $context = []): void
    {
        $this->log('info', $message, $context);
    }

    public function debug(\Stringable|string $message, array $context = []): void
    {
        $this->log('debug', $message, $context);
    }

    public function log($level, \Stringable|string $message, array $context = []): void
    {
        $this->records[] = [
            'level' => (string) $level,
            'message' => (string) $message,
            'context' => $context,
        ];
    }
}
