<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Support;

final class EnvironmentChecker
{
    public static function inspectForLlamaCpp(string $libraryPath): EnvironmentReport
    {
        $issues = [];

        if (!extension_loaded('FFI')) {
            $issues[] = new EnvironmentIssue('error', 'The PHP FFI extension is not loaded.');
        }

        $ffiEnable = strtolower((string) ini_get('ffi.enable'));
        if (!in_array($ffiEnable, ['1', 'on', 'true'], true)) {
            $issues[] = new EnvironmentIssue(
                'error',
                sprintf('`ffi.enable` must be enabled for runtime bindings. Current value: "%s".', $ffiEnable === '' ? 'empty' : $ffiEnable),
            );
        }

        if (!is_file($libraryPath)) {
            $issues[] = new EnvironmentIssue('error', sprintf('Native library not found at "%s".', $libraryPath));
        }

        if (!in_array(PHP_OS_FAMILY, ['Darwin', 'Linux'], true)) {
            $issues[] = new EnvironmentIssue(
                'warning',
                sprintf('Validated hosts are macOS and Linux. Current OS family: %s.', PHP_OS_FAMILY),
            );
        }

        $machine = php_uname('m');
        if (PHP_OS_FAMILY === 'Darwin' && !self::matchesArchitecture($machine, ['arm64', 'aarch64'])) {
            $issues[] = new EnvironmentIssue(
                'warning',
                sprintf('macOS validation currently targets Apple Silicon. Current machine architecture: %s.', $machine),
            );
        }

        if (PHP_OS_FAMILY === 'Linux' && !self::matchesArchitecture($machine, ['x86_64', 'amd64', 'arm64', 'aarch64'])) {
            $issues[] = new EnvironmentIssue(
                'warning',
                sprintf('Linux validation currently covers x86_64 and arm64 hosts. Current machine architecture: %s.', $machine),
            );
        }

        $loaderEnv = RuntimePlatform::loaderPathEnvironmentVariable();
        $libraryDir = dirname($libraryPath);
        if (
            $loaderEnv !== null
            && is_file($libraryPath)
            && !self::containsPath(getenv($loaderEnv), $libraryDir)
        ) {
            $issues[] = new EnvironmentIssue(
                'info',
                sprintf(
                    '`%s` does not include "%s". The default build embeds a relative runtime search path, but custom builds may still require this environment variable for sibling `ggml` libraries.',
                    $loaderEnv,
                    $libraryDir,
                ),
            );
        }

        if (extension_loaded('xdebug') || extension_loaded('pcov')) {
            $issues[] = new EnvironmentIssue(
                'warning',
                'Debugging or coverage extensions are loaded. Disable them for cleaner native timings and fewer surprises under heavy FFI workloads.',
            );
        }

        return new EnvironmentReport($libraryPath, $issues);
    }

    /**
     * @param list<string> $architectures
     */
    private static function matchesArchitecture(string $machine, array $architectures): bool
    {
        return in_array(strtolower($machine), array_map('strtolower', $architectures), true);
    }

    private static function containsPath(string|false $pathList, string $candidate): bool
    {
        if (!is_string($pathList) || $pathList === '') {
            return false;
        }

        foreach (explode(PATH_SEPARATOR, $pathList) as $path) {
            if (rtrim($path, DIRECTORY_SEPARATOR) === rtrim($candidate, DIRECTORY_SEPARATOR)) {
                return true;
            }
        }

        return false;
    }
}
