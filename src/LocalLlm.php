<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm;

use HelgeSverre\LocalLlm\Backend\SessionOptions;
use HelgeSverre\LocalLlm\Chat\ChatFormatterInterface;
use HelgeSverre\LocalLlm\FFI\LlamaBackend;
use HelgeSverre\LocalLlm\FFI\LlamaBackendConfig;
use HelgeSverre\LocalLlm\Runtime\LlamaRuntime;
use HelgeSverre\LocalLlm\Support\AppleSiliconTier;
use HelgeSverre\LocalLlm\Support\EnvironmentChecker;
use HelgeSverre\LocalLlm\Support\EnvironmentReport;
use HelgeSverre\LocalLlm\Support\RuntimePlatform;
use HelgeSverre\LocalLlm\Support\SupportedModelCatalog;
use HelgeSverre\LocalLlm\Support\SupportedModelProfile;
use Psr\Log\LoggerInterface;
use Psr\Log\LogLevel;

final class LocalLlm
{
    public static function llamaCpp(?LlamaBackendConfig $config = null): LlamaBackend
    {
        return new LlamaBackend($config);
    }

    public static function llamaCppRuntime(
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
    ): LlamaRuntime {
        return LlamaRuntime::fromModelPath(
            modelPath: $modelPath,
            libraryPath: $libraryPath,
            gpuLayers: $gpuLayers ?? RuntimePlatform::defaultGpuLayers(),
            sessionOptions: $sessionOptions,
            multimodalProjectorPath: $multimodalProjectorPath,
            multimodalProjectorUseGpu: $multimodalProjectorUseGpu,
            chatFormatter: $chatFormatter,
            logger: $logger,
            captureNativeLogs: $captureNativeLogs,
            nativeLogLevel: $nativeLogLevel,
        );
    }

    public static function llamaCppRuntimeFromOllama(
        string $ollamaModel,
        ?string $libraryPath = null,
        ?int $gpuLayers = null,
        ?SessionOptions $sessionOptions = null,
        ?ChatFormatterInterface $chatFormatter = null,
        ?LoggerInterface $logger = null,
        bool $captureNativeLogs = true,
        string $nativeLogLevel = LogLevel::WARNING,
        ?string $multimodalProjectorPath = null,
        bool $multimodalProjectorUseGpu = true,
    ): LlamaRuntime {
        return LlamaRuntime::fromOllamaModel(
            ollamaModel: $ollamaModel,
            libraryPath: $libraryPath,
            gpuLayers: $gpuLayers ?? RuntimePlatform::defaultGpuLayers(),
            sessionOptions: $sessionOptions,
            multimodalProjectorPath: $multimodalProjectorPath,
            multimodalProjectorUseGpu: $multimodalProjectorUseGpu,
            chatFormatter: $chatFormatter,
            logger: $logger,
            captureNativeLogs: $captureNativeLogs,
            nativeLogLevel: $nativeLogLevel,
        );
    }

    public static function inspectLlamaCppEnvironment(?string $libraryPath = null): EnvironmentReport
    {
        return EnvironmentChecker::inspectForLlamaCpp($libraryPath ?? LlamaBackend::defaultLibraryPath());
    }

    /**
     * @return list<SupportedModelProfile>
     */
    public static function supportedModels(): array
    {
        return SupportedModelCatalog::all();
    }

    public static function supportedModel(string $id): SupportedModelProfile
    {
        return SupportedModelCatalog::get($id);
    }

    public static function recommendedSessionOptions(string $modelId, AppleSiliconTier $tier): SessionOptions
    {
        return self::supportedModel($modelId)->recommendedSessionOptions($tier);
    }
}
