<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\FFI\LlamaNativeLogSink;
use PHPUnit\Framework\TestCase;
use Psr\Log\LoggerInterface;
use Psr\Log\LogLevel;

final class LlamaNativeLogSinkTest extends TestCase
{
    public function testItFlushesContinuedNativeMessagesAsOneRecord(): void
    {
        $logger = new RecordingLogger();
        $sink = new LlamaNativeLogSink($logger, LogLevel::DEBUG);

        $sink->consume(3, "main line");
        $sink->consume(5, " continued\n");

        self::assertCount(1, $logger->records);
        self::assertSame(LogLevel::WARNING, $logger->records[0]['level']);
        self::assertSame('main line continued', $logger->records[0]['message']);
        self::assertSame('llama.cpp-native', $logger->records[0]['context']['source']);
    }

    public function testItFiltersMessagesBelowTheConfiguredMinimumLevel(): void
    {
        $logger = new RecordingLogger();
        $sink = new LlamaNativeLogSink($logger, LogLevel::ERROR);

        $sink->consume(2, "informational line\n");
        $sink->consume(4, "error line\n");

        self::assertCount(1, $logger->records);
        self::assertSame(LogLevel::ERROR, $logger->records[0]['level']);
        self::assertSame('error line', $logger->records[0]['message']);
    }
}

final class RecordingLogger implements LoggerInterface
{
    /** @var list<array{level:string,message:string,context:array<mixed>}> */
    public array $records = [];

    public function emergency(\Stringable|string $message, array $context = []): void
    {
        $this->log(LogLevel::EMERGENCY, $message, $context);
    }

    public function alert(\Stringable|string $message, array $context = []): void
    {
        $this->log(LogLevel::ALERT, $message, $context);
    }

    public function critical(\Stringable|string $message, array $context = []): void
    {
        $this->log(LogLevel::CRITICAL, $message, $context);
    }

    public function error(\Stringable|string $message, array $context = []): void
    {
        $this->log(LogLevel::ERROR, $message, $context);
    }

    public function warning(\Stringable|string $message, array $context = []): void
    {
        $this->log(LogLevel::WARNING, $message, $context);
    }

    public function notice(\Stringable|string $message, array $context = []): void
    {
        $this->log(LogLevel::NOTICE, $message, $context);
    }

    public function info(\Stringable|string $message, array $context = []): void
    {
        $this->log(LogLevel::INFO, $message, $context);
    }

    public function debug(\Stringable|string $message, array $context = []): void
    {
        $this->log(LogLevel::DEBUG, $message, $context);
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
