<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\FFI;

use FFI\CData;

final class LlamaSingleTokenBatchBuffer
{
    private CData $tokenBuffer;

    public function __construct(
        private readonly LlamaLibrary $library,
    ) {
        $this->tokenBuffer = $library->new('llama_token[1]');
    }

    public function batchFor(int $tokenId): CData
    {
        $this->tokenBuffer[0] = $tokenId;

        return $this->library->ffi()->llama_batch_get_one($this->tokenBuffer, 1);
    }
}
