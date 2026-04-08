<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\Support\OllamaModelResolver;
use PHPUnit\Framework\TestCase;

final class OllamaModelResolverTest extends TestCase
{
    public function testItParsesAnAbsoluteBlobPathFromAModelfile(): void
    {
        $modelfile = <<<TEXT
        FROM /Users/example/.ollama/models/blobs/sha256-abc123
        TEMPLATE """{{ .Prompt }}"""
        TEXT;

        self::assertSame(
            '/Users/example/.ollama/models/blobs/sha256-abc123',
            OllamaModelResolver::parseModelfile($modelfile),
        );
    }

    public function testItReturnsNullWhenTheModelfileDoesNotPointAtALocalPath(): void
    {
        $modelfile = <<<TEXT
        FROM llama3.2
        PARAMETER temperature 0
        TEXT;

        self::assertNull(OllamaModelResolver::parseModelfile($modelfile));
    }
}
