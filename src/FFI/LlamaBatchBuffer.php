<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\FFI;

use FFI\CData;

final class LlamaBatchBuffer
{
    private CData $tokenBuffer;

    /**
     * @param list<int> $tokenIds
     */
    public function __construct(
        private readonly LlamaLibrary $library,
        private readonly array $tokenIds,
    ) {
        $count = count($tokenIds);
        $this->tokenBuffer = $library->new(sprintf('llama_token[%d]', max($count, 1)));

        foreach ($tokenIds as $index => $tokenId) {
            $this->tokenBuffer[$index] = $tokenId;
        }
    }

    public function batch(): CData
    {
        return $this->library->ffi()->llama_batch_get_one($this->tokenBuffer, count($this->tokenIds));
    }
}
