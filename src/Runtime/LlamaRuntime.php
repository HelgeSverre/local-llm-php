<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Runtime;

use HelgeSverre\LocalLlm\Backend\ModelInterface;
use HelgeSverre\LocalLlm\Backend\ModelMetadata;
use HelgeSverre\LocalLlm\Backend\ModelOptions;
use HelgeSverre\LocalLlm\Backend\SessionInterface;
use HelgeSverre\LocalLlm\Backend\SessionOptions;
use HelgeSverre\LocalLlm\Chat\ChatFormatterInterface;
use HelgeSverre\LocalLlm\Chat\ChatSession;
use HelgeSverre\LocalLlm\Chat\DefaultChatFormatter;
use HelgeSverre\LocalLlm\FFI\LlamaBackend;
use HelgeSverre\LocalLlm\FFI\LlamaBackendConfig;
use HelgeSverre\LocalLlm\Support\EnvironmentChecker;
use HelgeSverre\LocalLlm\Support\EnvironmentReport;
use HelgeSverre\LocalLlm\Support\OllamaModelResolver;
use HelgeSverre\LocalLlm\Support\RuntimePlatform;
use Psr\Log\LoggerInterface;
use Psr\Log\LogLevel;
use Psr\Log\NullLogger;

final class LlamaRuntime
{
    private bool $closed = false;

    private function __construct(
        private readonly ModelInterface $model,
        private readonly SessionOptions $defaultSessionOptions,
        private readonly EnvironmentReport $environmentReport,
        private readonly string $resolvedModelPath,
        private readonly ChatFormatterInterface $defaultChatFormatter,
        private readonly LoggerInterface $logger,
    ) {}

    public static function fromModelPath(
        string $modelPath,
        ?string $libraryPath = null,
        ?int $gpuLayers = null,
        ?SessionOptions $sessionOptions = null,
        ?ChatFormatterInterface $chatFormatter = null,
        ?LoggerInterface $logger = null,
        bool $captureNativeLogs = true,
        string $nativeLogLevel = LogLevel::WARNING,
        ?string $multimodalProjectorPath = null,
        bool $multimodalProjectorUseGpu = true,
    ): self {
        $libraryPath ??= $multimodalProjectorPath !== null
            ? LlamaBackend::defaultMultimodalLibraryPath()
            : LlamaBackend::defaultLibraryPath();

        if (
            $multimodalProjectorPath !== null
            && basename($libraryPath) === RuntimePlatform::sharedLibraryBasename()
        ) {
            $candidateMtmdPath = dirname($libraryPath) . '/' . RuntimePlatform::multimodalSharedLibraryBasename();
            if (is_file($candidateMtmdPath)) {
                $libraryPath = $candidateMtmdPath;
            }
        }

        $logger ??= new NullLogger();
        $gpuLayers ??= RuntimePlatform::defaultGpuLayers();
        $environmentReport = EnvironmentChecker::inspectForLlamaCpp($libraryPath);
        $environmentReport->assertReady();

        $backend = new LlamaBackend(new LlamaBackendConfig(
            $libraryPath,
            logger: $logger,
            captureNativeLogs: $captureNativeLogs,
            nativeLogLevel: $nativeLogLevel,
        ));
        $model = $backend->loadModel(new ModelOptions(
            modelPath: $modelPath,
            gpuLayers: $gpuLayers,
            multimodalProjectorPath: $multimodalProjectorPath,
            multimodalProjectorUseGpu: $multimodalProjectorUseGpu,
        ));

        return new self(
            model: $model,
            defaultSessionOptions: $sessionOptions ?? new SessionOptions(),
            environmentReport: $environmentReport,
            resolvedModelPath: $modelPath,
            defaultChatFormatter: $chatFormatter ?? new DefaultChatFormatter(),
            logger: $logger,
        );
    }

    public static function fromOllamaModel(
        string $ollamaModel,
        ?string $libraryPath = null,
        ?int $gpuLayers = null,
        ?SessionOptions $sessionOptions = null,
        ?OllamaModelResolver $resolver = null,
        ?ChatFormatterInterface $chatFormatter = null,
        ?LoggerInterface $logger = null,
        bool $captureNativeLogs = true,
        string $nativeLogLevel = LogLevel::WARNING,
        ?string $multimodalProjectorPath = null,
        bool $multimodalProjectorUseGpu = true,
    ): self {
        $resolver ??= new OllamaModelResolver();
        $modelPath = $resolver->resolveBlobPath($ollamaModel);

        return self::fromModelPath(
            modelPath: $modelPath,
            libraryPath: $libraryPath,
            gpuLayers: $gpuLayers,
            sessionOptions: $sessionOptions,
            multimodalProjectorPath: $multimodalProjectorPath,
            multimodalProjectorUseGpu: $multimodalProjectorUseGpu,
            chatFormatter: $chatFormatter,
            logger: $logger,
            captureNativeLogs: $captureNativeLogs,
            nativeLogLevel: $nativeLogLevel,
        );
    }

    public function environmentReport(): EnvironmentReport
    {
        return $this->environmentReport;
    }

    public function resolvedModelPath(): string
    {
        return $this->resolvedModelPath;
    }

    public function metadata(): ModelMetadata
    {
        $this->assertOpen();

        return $this->model->metadata();
    }

    public function model(): ModelInterface
    {
        $this->assertOpen();

        return $this->model;
    }

    public function newSession(?SessionOptions $options = null): SessionInterface
    {
        $this->assertOpen();

        return $this->model->createSession($options ?? $this->defaultSessionOptions);
    }

    public function newChatSession(
        ?SessionOptions $options = null,
        ?ChatFormatterInterface $formatter = null,
    ): ChatSession {
        $this->assertOpen();

        return new ChatSession(
            model: $this->model,
            session: $this->newSession($options),
            formatter: $formatter ?? $this->defaultChatFormatter,
            logger: $this->logger,
        );
    }

    public function close(): void
    {
        if ($this->closed) {
            return;
        }

        $this->model->close();
        $this->closed = true;
    }

    public function __destruct()
    {
        $this->close();
    }

    private function assertOpen(): void
    {
        if ($this->closed) {
            throw new \RuntimeException('Runtime has already been closed.');
        }
    }
}
