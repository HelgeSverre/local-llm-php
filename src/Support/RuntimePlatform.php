<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Support;

final class RuntimePlatform
{
    public static function sharedLibraryBasename(?string $osFamily = null): string
    {
        return 'libllama.' . self::sharedLibraryExtension($osFamily);
    }

    public static function multimodalSharedLibraryBasename(?string $osFamily = null): string
    {
        return 'libmtmd.' . self::sharedLibraryExtension($osFamily);
    }

    public static function sharedLibraryExtension(?string $osFamily = null): string
    {
        return match (self::osFamily($osFamily)) {
            'Darwin' => 'dylib',
            default => 'so',
        };
    }

    public static function loaderPathEnvironmentVariable(?string $osFamily = null): ?string
    {
        return match (self::osFamily($osFamily)) {
            'Darwin' => 'DYLD_LIBRARY_PATH',
            'Linux' => 'LD_LIBRARY_PATH',
            default => null,
        };
    }

    public static function defaultGpuLayers(?string $osFamily = null): int
    {
        return self::osFamily($osFamily) === 'Darwin' ? 99 : 0;
    }

    public static function osFamily(?string $osFamily = null): string
    {
        return $osFamily ?? PHP_OS_FAMILY;
    }
}
