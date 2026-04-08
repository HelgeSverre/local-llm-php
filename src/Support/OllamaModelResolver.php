<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Support;

use HelgeSverre\LocalLlm\Backend\BackendException;

final readonly class OllamaModelResolver
{
    public function __construct(
        private string $ollamaBinary = 'ollama',
    ) {
    }

    public function resolveBlobPath(string $modelName): string
    {
        $modelfile = $this->fetchModelfile($modelName);
        $path = self::parseModelfile($modelfile);

        if ($path === null || !is_file($path)) {
            throw new BackendException(sprintf(
                'Could not resolve a local GGUF blob path for Ollama model "%s".',
                $modelName,
            ));
        }

        return $path;
    }

    public function fetchModelfile(string $modelName): string
    {
        $command = sprintf(
            '%s show --modelfile %s 2>/dev/null',
            escapeshellcmd($this->ollamaBinary),
            escapeshellarg($modelName),
        );

        $output = shell_exec($command);
        if (!is_string($output) || trim($output) === '') {
            throw new BackendException(sprintf('Failed to inspect local Ollama model "%s".', $modelName));
        }

        return $output;
    }

    public static function parseModelfile(string $modelfile): ?string
    {
        foreach (preg_split('/\R/', $modelfile) ?: [] as $line) {
            $trimmed = trim($line);
            if (!str_starts_with($trimmed, 'FROM ')) {
                continue;
            }

            $candidate = trim(substr($trimmed, strlen('FROM ')));
            if ($candidate === '') {
                continue;
            }

            if (str_starts_with($candidate, '/') || str_starts_with($candidate, './') || str_starts_with($candidate, '../')) {
                return $candidate;
            }
        }

        return null;
    }
}
