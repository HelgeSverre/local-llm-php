<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Backend;

use Generator;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\Generation\GenerationResult;
use HelgeSverre\LocalLlm\Generation\PromptEvaluationResult;
use HelgeSverre\LocalLlm\Generation\SessionState;
use HelgeSverre\LocalLlm\Generation\TokenChunk;

interface SessionInterface
{
    /**
     * @param string|list<int> $prompt
     */
    public function evaluate(string|array $prompt, bool $addSpecial = true, bool $parseSpecial = true): PromptEvaluationResult;

    /**
     * @param callable(TokenChunk): void|null $onToken
     */
    public function generate(GenerationConfig $config, ?callable $onToken = null): GenerationResult;

    /**
     * @return Generator<int, TokenChunk>
     */
    public function stream(GenerationConfig $config): Generator;

    public function snapshot(): SessionState;

    public function restore(SessionState $state): void;

    public function reset(bool $clearStateData = true): void;

    public function close(): void;
}
