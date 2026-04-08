<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Backend;

use HelgeSverre\LocalLlm\Tokenizer\TokenizationResult;

interface ModelInterface
{
    public function backendName(): string;

    public function metadata(): ModelMetadata;

    public function tokenize(string $text, bool $addSpecial = true, bool $parseSpecial = true): TokenizationResult;

    /**
     * @param list<int> $tokens
     */
    public function detokenize(array $tokens, bool $removeSpecial = false, bool $unparseSpecial = true): string;

    public function createSession(?SessionOptions $options = null): SessionInterface;

    public function close(): void;
}
