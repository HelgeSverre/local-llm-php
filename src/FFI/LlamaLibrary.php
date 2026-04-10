<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\FFI;

use FFI;
use FFI\CData;
use HelgeSverre\LocalLlm\Backend\BackendException;
use HelgeSverre\LocalLlm\Support\RuntimePlatform;
use Psr\Log\LoggerInterface;

final class LlamaLibrary
{
    private static bool $backendInitialized = false;
    private static mixed $nativeLogCallback = null;
    private static ?LlamaNativeLogSink $nativeLogSink = null;

    private FFI $ffi;

    public function __construct(private readonly LlamaBackendConfig $config)
    {
        if (!is_file($config->libraryPath)) {
            throw new BackendException(sprintf('llama.cpp library not found at "%s".', $config->libraryPath));
        }

        $ffiLibraryPath = $this->resolveFfiLibraryPath($config->libraryPath);

        $config->logger->debug('Binding llama.cpp FFI library.', [
            'library_path' => $config->libraryPath,
            'ffi_library_path' => $ffiLibraryPath,
            'load_dynamic_backends' => $config->loadDynamicBackends,
            'initialize_backend' => $config->initializeBackend,
        ]);

        $this->ffi = FFI::cdef(LlamaCdef::definitions(), $ffiLibraryPath);
        $this->installNativeLogRouting();

        if ($config->loadDynamicBackends && $this->hasFunction('ggml_backend_load_all')) {
            $this->ffi->ggml_backend_load_all();
        }

        if ($config->initializeBackend && !self::$backendInitialized) {
            $this->ffi->llama_backend_init();
            self::$backendInitialized = true;
            $config->logger->debug('Initialized llama.cpp backend.');
        }
    }

    public function ffi(): FFI
    {
        return $this->ffi;
    }

    public function new(string $type, bool $owned = true): CData
    {
        return $this->ffi->new($type, $owned);
    }

    public function cast(string $type, CData $value): CData
    {
        return $this->ffi->cast($type, $value);
    }

    public function string(CData $value, ?int $size = null): string
    {
        return FFI::string($value, $size);
    }

    public function memcpy(CData $to, string $from, int $length): void
    {
        FFI::memcpy($to, $from, $length);
    }

    public function addr(CData $value): CData
    {
        return FFI::addr($value);
    }

    public function isNull(mixed $value): bool
    {
        if ($value === null) {
            return true;
        }

        if (!$value instanceof CData) {
            return false;
        }

        return FFI::isNull($value);
    }

    public function logger(): LoggerInterface
    {
        return $this->config->logger;
    }

    public function __destruct()
    {
        self::$nativeLogSink?->flush();
    }

    private function installNativeLogRouting(): void
    {
        if (!$this->config->captureNativeLogs || !$this->hasFunction('llama_log_set')) {
            return;
        }

        self::$nativeLogSink = new LlamaNativeLogSink($this->config->logger, $this->config->nativeLogLevel);
        self::$nativeLogCallback ??= static function (int $level, mixed $text, mixed $userData): void {
            if (self::$nativeLogSink === null) {
                return;
            }

            if (is_string($text)) {
                self::$nativeLogSink->consume($level, $text);

                return;
            }

            if ($text instanceof CData && !FFI::isNull($text)) {
                self::$nativeLogSink->consume($level, FFI::string($text));
            }
        };

        $this->ffi->llama_log_set(self::$nativeLogCallback, null);
        $this->config->logger->debug('Installed llama.cpp native log routing.', [
            'native_log_level' => $this->config->nativeLogLevel,
        ]);
    }

    private function hasFunction(string $function): bool
    {
        try {
            $this->ffi->$function;

            return true;
        } catch (\Throwable) {
            return false;
        }
    }

    private function resolveFfiLibraryPath(string $libraryPath): string
    {
        if (basename($libraryPath) === RuntimePlatform::multimodalSharedLibraryBasename()) {
            return $libraryPath;
        }

        $candidate = dirname($libraryPath) . '/' . RuntimePlatform::multimodalSharedLibraryBasename();

        return is_file($candidate) ? $candidate : $libraryPath;
    }
}
