<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\FFI;

use Psr\Log\LoggerInterface;
use Psr\Log\LogLevel;

final class LlamaNativeLogSink
{
    private string $buffer = '';
    private int $bufferLevel = self::LEVEL_INFO;

    private const LEVEL_NONE = 0;
    private const LEVEL_DEBUG = 1;
    private const LEVEL_INFO = 2;
    private const LEVEL_WARN = 3;
    private const LEVEL_ERROR = 4;
    private const LEVEL_CONT = 5;

    public function __construct(
        private readonly LoggerInterface $logger,
        private readonly string $minimumLevel = LogLevel::WARNING,
    ) {}

    public function consume(int $level, string $text): void
    {
        if ($text === '') {
            return;
        }

        if ($level !== self::LEVEL_CONT && $this->buffer !== '') {
            $this->flush();
        }

        if ($level !== self::LEVEL_CONT) {
            $this->bufferLevel = $level;
        }

        $this->buffer .= $text;
        $this->flushCompleteLines();
    }

    public function flush(): void
    {
        if ($this->buffer === '') {
            return;
        }

        $message = rtrim($this->buffer, "\r\n");
        $this->buffer = '';

        if ($message === '') {
            return;
        }

        $level = $this->mapLevel($this->bufferLevel);
        if ($this->severityRank($level) < $this->severityRank($this->minimumLevel)) {
            return;
        }

        $this->logger->log($level, $message, [
            'source' => 'llama.cpp-native',
            'native_level' => $this->bufferLevel,
        ]);
    }

    private function flushCompleteLines(): void
    {
        while (($newline = strpos($this->buffer, "\n")) !== false) {
            $line = substr($this->buffer, 0, $newline + 1);
            $this->buffer = (string) substr($this->buffer, $newline + 1);

            $message = rtrim($line, "\r\n");
            if ($message === '') {
                continue;
            }

            $level = $this->mapLevel($this->bufferLevel);
            if ($this->severityRank($level) < $this->severityRank($this->minimumLevel)) {
                continue;
            }

            $this->logger->log($level, $message, [
                'source' => 'llama.cpp-native',
                'native_level' => $this->bufferLevel,
            ]);
        }
    }

    private function mapLevel(int $level): string
    {
        return match ($level) {
            self::LEVEL_ERROR => LogLevel::ERROR,
            self::LEVEL_WARN => LogLevel::WARNING,
            self::LEVEL_DEBUG => LogLevel::DEBUG,
            self::LEVEL_NONE => LogLevel::DEBUG,
            default => LogLevel::INFO,
        };
    }

    private function severityRank(string $level): int
    {
        return match ($level) {
            LogLevel::DEBUG => 100,
            LogLevel::INFO => 200,
            LogLevel::NOTICE => 250,
            LogLevel::WARNING => 300,
            LogLevel::ERROR => 400,
            LogLevel::CRITICAL => 500,
            LogLevel::ALERT => 600,
            LogLevel::EMERGENCY => 700,
            default => 0,
        };
    }
}
