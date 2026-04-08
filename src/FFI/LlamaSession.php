<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\FFI;

use FFI\CData;
use Generator;
use HelgeSverre\LocalLlm\Backend\BackendException;
use HelgeSverre\LocalLlm\Backend\SessionInterface;
use HelgeSverre\LocalLlm\Backend\SessionOptions;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\Generation\GenerationProfile;
use HelgeSverre\LocalLlm\Generation\GenerationResult;
use HelgeSverre\LocalLlm\Generation\PromptEvaluationResult;
use HelgeSverre\LocalLlm\Generation\SessionState;
use HelgeSverre\LocalLlm\Generation\TokenChunk;

final class LlamaSession implements SessionInterface
{
    private const FINISH_MAX_TOKENS = 'max_tokens';
    private const FINISH_STOP_TOKEN = 'stop_token';
    private const FINISH_STOP_STRING = 'stop_string';
    private const FINISH_EOG = 'end_of_generation';

    private CData $context;
    private bool $closed = false;

    /** @var list<int> */
    private array $historyTokenIds = [];

    private ?PromptEvaluationResult $lastPromptEvaluation = null;

    public function __construct(
        private readonly LlamaLibrary $library,
        private readonly LlamaModel $model,
        private readonly SessionOptions $options,
    ) {
        $ffi = $library->ffi();
        $params = $ffi->llama_context_default_params();
        $params->n_ctx = $options->contextTokens;
        $params->n_batch = $options->batchSize;
        $params->n_ubatch = $options->microBatchSize;
        $params->n_seq_max = $options->maxSequences;
        $params->n_threads = $options->threads;
        $params->n_threads_batch = $options->batchThreads;
        $params->flash_attn_type = $options->flashAttention ? 1 : 0;
        $params->offload_kqv = $options->offloadKqv;
        $params->embeddings = $options->embeddings;
        $params->no_perf = false;

        $this->context = $ffi->llama_init_from_model($model->modelHandle(), $params);
        if ($library->isNull($this->context)) {
            throw new BackendException('Failed to initialize llama.cpp context.');
        }

        if ($options->threads > 0 || $options->batchThreads > 0) {
            $ffi->llama_set_n_threads($this->context, $options->threads, $options->batchThreads);
        }

        $this->library->logger()->debug('Created llama.cpp session.', [
            'context_tokens' => $options->contextTokens,
            'batch_size' => $options->batchSize,
            'micro_batch_size' => $options->microBatchSize,
            'flash_attention' => $options->flashAttention,
            'offload_kqv' => $options->offloadKqv,
        ]);
    }

    public function __destruct()
    {
        $this->close();
    }

    public function evaluate(string|array $prompt, bool $addSpecial = true, bool $parseSpecial = true): PromptEvaluationResult
    {
        $this->assertOpen();

        $tokenIds = is_string($prompt)
            ? $this->model->tokenize($prompt, $addSpecial, $parseSpecial)->tokens
            : array_values($prompt);

        if ($tokenIds === []) {
            $result = new PromptEvaluationResult([], 0, 0);
            $this->lastPromptEvaluation = $result;

            return $result;
        }

        $ffi = $this->library->ffi();
        $start = hrtime(true);
        $ffi->llama_set_warmup($this->context, false);
        $hasEncoder = $ffi->llama_model_has_encoder($this->model->modelHandle());
        $chunkCount = 0;

        foreach (array_chunk($tokenIds, max(1, $this->options->batchSize)) as $tokenChunk) {
            $chunkCount++;
            $batchBuffer = new LlamaBatchBuffer($this->library, $tokenChunk);
            $batch = $batchBuffer->batch();

            $status = $hasEncoder
                ? $ffi->llama_encode($this->context, $batch)
                : $ffi->llama_decode($this->context, $batch);

            if ($status !== 0) {
                throw new BackendException(sprintf('Prompt evaluation failed with status %d.', $status));
            }
        }

        if ($hasEncoder) {
            $decoderStartToken = (int) $ffi->llama_model_decoder_start_token($this->model->modelHandle());
            if ($decoderStartToken === -1) {
                $decoderStartToken = (int) $ffi->llama_vocab_bos($this->model->vocabHandle());
            }

            $decoderBatchBuffer = new LlamaBatchBuffer($this->library, [$decoderStartToken]);
            $decoderBatch = $decoderBatchBuffer->batch();
            $decodeStatus = $ffi->llama_decode($this->context, $decoderBatch);
            if ($decodeStatus !== 0) {
                throw new BackendException(sprintf('Decoder start evaluation failed with status %d.', $decodeStatus));
            }

            $this->historyTokenIds[] = $decoderStartToken;
        }

        $durationUs = (int) ((hrtime(true) - $start) / 1_000);
        array_push($this->historyTokenIds, ...$tokenIds);

        $this->library->logger()->debug('Evaluated prompt.', [
            'prompt_tokens' => count($tokenIds),
            'chunks' => $chunkCount,
            'duration_us' => $durationUs,
            'chunked' => $chunkCount > 1,
        ]);

        $result = new PromptEvaluationResult($tokenIds, count($tokenIds), $durationUs);
        $this->lastPromptEvaluation = $result;

        return $result;
    }

    public function generate(GenerationConfig $config, ?callable $onToken = null): GenerationResult
    {
        $this->assertOpen();

        $ffi = $this->library->ffi();
        $ffi->llama_perf_context_reset($this->context);

        $promptEvaluation = null;
        if ($config->prompt !== null) {
            $promptEvaluation = $this->evaluate($config->prompt, $config->addSpecial, $config->parseSpecial);
        } else {
            $this->lastPromptEvaluation = null;
        }

        if ($this->historyTokenIds === []) {
            throw new BackendException('Generation requires a prompt or a previously evaluated session state.');
        }

        $sampler = $this->buildSampler($config);
        try {
            $generatedText = '';
            $generatedTokenIds = [];
            $chunks = [];
            $start = hrtime(true);
            $decodeStart = hrtime(true);
            $finishReason = self::FINISH_MAX_TOKENS;
            $decodeBatchBuffer = new LlamaSingleTokenBatchBuffer($this->library);
            $pieceBufferCapacity = 256;
            $pieceBuffer = $this->library->new(sprintf('char[%d]', $pieceBufferCapacity));
            $hasStopTokens = $config->stopTokens !== [];
            $hasStopStrings = $config->stopStrings !== [];
            $stopTokenLookup = $hasStopTokens ? array_fill_keys($config->stopTokens, true) : [];

            for ($index = 0; $index < $config->maxTokens; $index++) {
                $tokenId = (int) $ffi->llama_sampler_sample($sampler, $this->context, -1);
                if ($ffi->llama_vocab_is_eog($this->model->vocabHandle(), $tokenId)) {
                    $finishReason = self::FINISH_EOG;
                    break;
                }

                if ($hasStopTokens && isset($stopTokenLookup[$tokenId])) {
                    $finishReason = self::FINISH_STOP_TOKEN;
                    break;
                }

                $piece = $this->pieceForToken($tokenId, $config->unparseSpecial, $pieceBuffer, $pieceBufferCapacity);
                $generatedText .= $piece;
                $generatedTokenIds[] = $tokenId;
                $this->historyTokenIds[] = $tokenId;

                $chunk = new TokenChunk($tokenId, $piece, $index, (int) ((hrtime(true) - $start) / 1_000));
                $chunks[] = $chunk;
                if ($onToken !== null) {
                    $onToken($chunk);
                }

                if ($hasStopStrings && $this->endsWithStopString($generatedText, $config->stopStrings)) {
                    $generatedText = $this->trimStopStrings($generatedText, $config->stopStrings);
                    $finishReason = self::FINISH_STOP_STRING;
                    break;
                }

                $batch = $decodeBatchBuffer->batchFor($tokenId);
                $status = $ffi->llama_decode($this->context, $batch);
                if ($status !== 0) {
                    throw new BackendException(sprintf('Token decode failed with status %d.', $status));
                }
            }

            $decodeDurationUs = (int) ((hrtime(true) - $decodeStart) / 1_000);
            $totalDurationUs = (int) ((hrtime(true) - $start) / 1_000);
            $nativeContext = $ffi->llama_perf_context($this->context);
            $nativeSampler = $ffi->llama_perf_sampler($sampler);
            $profile = new GenerationProfile(
                promptTokens: $promptEvaluation?->tokenCount ?? 0,
                generatedTokens: count($generatedTokenIds),
                promptEvalDurationUs: $promptEvaluation?->durationUs ?? 0,
                decodeDurationUs: $decodeDurationUs,
                totalDurationUs: $totalDurationUs,
                nativePromptEvalMs: (float) $nativeContext->t_p_eval_ms,
                nativeDecodeMs: (float) $nativeContext->t_eval_ms,
                nativeSampleMs: (float) $nativeSampler->t_sample_ms,
                nativeGraphReuseCount: (int) $nativeContext->n_reused,
            );

            $this->library->logger()->info('Completed generation.', [
                'finish_reason' => $finishReason,
                'prompt_tokens' => $promptEvaluation?->tokenCount ?? 0,
                'generated_tokens' => count($generatedTokenIds),
                'decode_duration_us' => $decodeDurationUs,
                'total_duration_us' => $totalDurationUs,
                'native_decode_ms' => (float) $nativeContext->t_eval_ms,
                'native_prompt_eval_ms' => (float) $nativeContext->t_p_eval_ms,
            ]);

            return new GenerationResult(
                text: $generatedText,
                tokenIds: $generatedTokenIds,
                chunks: $chunks,
                finishReason: $finishReason,
                promptEvaluation: $promptEvaluation,
                profile: $profile,
            );
        } finally {
            $ffi->llama_sampler_free($sampler);
        }
    }

    public function stream(GenerationConfig $config): Generator
    {
        $this->assertOpen();

        foreach ($this->runGeneration($config) as $event) {
            if ($event instanceof TokenChunk) {
                yield $event;
            }
        }
    }

    public function snapshot(): SessionState
    {
        $this->assertOpen();

        $ffi = $this->library->ffi();
        $size = (int) $ffi->llama_state_get_size($this->context);
        $buffer = $this->library->new(sprintf('uint8_t[%d]', max($size, 1)));
        $written = (int) $ffi->llama_state_get_data($this->context, $buffer, $size);

        if ($written !== $size) {
            throw new BackendException(sprintf('Expected %d bytes when snapshotting session state, received %d.', $size, $written));
        }

        return new SessionState(
            bytes: $this->library->string($this->library->cast('char *', $buffer), $written),
            historyTokenIds: $this->historyTokenIds,
        );
    }

    public function restore(SessionState $state): void
    {
        $this->assertOpen();
        $this->reset();

        $ffi = $this->library->ffi();
        $length = strlen($state->bytes);
        $buffer = $this->library->new(sprintf('uint8_t[%d]', max($length, 1)));
        if ($length > 0) {
            $this->library->memcpy($buffer, $state->bytes, $length);
        }

        $read = (int) $ffi->llama_state_set_data($this->context, $buffer, $length);
        if ($read !== $length) {
            throw new BackendException(sprintf('Expected to restore %d bytes of session state, read %d.', $length, $read));
        }

        $this->historyTokenIds = $state->historyTokenIds;
    }

    public function reset(bool $clearStateData = true): void
    {
        $this->assertOpen();

        $ffi = $this->library->ffi();
        $memory = $ffi->llama_get_memory($this->context);
        $ffi->llama_memory_clear($memory, $clearStateData);
        $ffi->llama_perf_context_reset($this->context);
        $this->historyTokenIds = [];
        $this->lastPromptEvaluation = null;
    }

    public function close(): void
    {
        if ($this->closed) {
            return;
        }

        $this->library->ffi()->llama_free($this->context);
        $this->closed = true;
    }

    /**
     * @return Generator<int, TokenChunk|array{text:string,token_ids:list<int>,finish_reason:string,prompt_eval:?PromptEvaluationResult,profile:GenerationProfile}>
     */
    private function runGeneration(GenerationConfig $config): Generator
    {
        $this->assertOpen();

        $ffi = $this->library->ffi();
        $ffi->llama_perf_context_reset($this->context);

        $promptEvaluation = null;
        if ($config->prompt !== null) {
            $promptEvaluation = $this->evaluate($config->prompt, $config->addSpecial, $config->parseSpecial);
        } else {
            $this->lastPromptEvaluation = null;
        }

        if ($this->historyTokenIds === []) {
            throw new BackendException('Generation requires a prompt or a previously evaluated session state.');
        }

        $sampler = $this->buildSampler($config);
        try {
            $generatedText = '';
            $generatedTokenIds = [];
            $start = hrtime(true);
            $decodeStart = hrtime(true);
            $finishReason = self::FINISH_MAX_TOKENS;
            $decodeBatchBuffer = new LlamaSingleTokenBatchBuffer($this->library);
            $pieceBufferCapacity = 256;
            $pieceBuffer = $this->library->new(sprintf('char[%d]', $pieceBufferCapacity));
            $hasStopTokens = $config->stopTokens !== [];
            $hasStopStrings = $config->stopStrings !== [];
            $stopTokenLookup = $hasStopTokens ? array_fill_keys($config->stopTokens, true) : [];

            for ($index = 0; $index < $config->maxTokens; $index++) {
                $tokenId = (int) $ffi->llama_sampler_sample($sampler, $this->context, -1);
                if ($ffi->llama_vocab_is_eog($this->model->vocabHandle(), $tokenId)) {
                    $finishReason = self::FINISH_EOG;
                    break;
                }

                if ($hasStopTokens && isset($stopTokenLookup[$tokenId])) {
                    $finishReason = self::FINISH_STOP_TOKEN;
                    break;
                }

                $piece = $this->pieceForToken($tokenId, $config->unparseSpecial, $pieceBuffer, $pieceBufferCapacity);
                $generatedText .= $piece;
                $generatedTokenIds[] = $tokenId;
                $this->historyTokenIds[] = $tokenId;

                if ($hasStopStrings && $this->endsWithStopString($generatedText, $config->stopStrings)) {
                    $generatedText = $this->trimStopStrings($generatedText, $config->stopStrings);
                    $finishReason = self::FINISH_STOP_STRING;
                    yield new TokenChunk($tokenId, $piece, $index, (int) ((hrtime(true) - $start) / 1_000));
                    break;
                }

                yield new TokenChunk($tokenId, $piece, $index, (int) ((hrtime(true) - $start) / 1_000));

                $batch = $decodeBatchBuffer->batchFor($tokenId);
                $status = $ffi->llama_decode($this->context, $batch);
                if ($status !== 0) {
                    throw new BackendException(sprintf('Token decode failed with status %d.', $status));
                }
            }

            $decodeDurationUs = (int) ((hrtime(true) - $decodeStart) / 1_000);
            $totalDurationUs = (int) ((hrtime(true) - $start) / 1_000);
            $nativeContext = $ffi->llama_perf_context($this->context);
            $nativeSampler = $ffi->llama_perf_sampler($sampler);

            yield [
                'text' => $generatedText,
                'token_ids' => $generatedTokenIds,
                'finish_reason' => $finishReason,
                'prompt_eval' => $promptEvaluation,
                'profile' => new GenerationProfile(
                    promptTokens: $promptEvaluation?->tokenCount ?? 0,
                    generatedTokens: count($generatedTokenIds),
                    promptEvalDurationUs: $promptEvaluation?->durationUs ?? 0,
                    decodeDurationUs: $decodeDurationUs,
                    totalDurationUs: $totalDurationUs,
                    nativePromptEvalMs: (float) $nativeContext->t_p_eval_ms,
                    nativeDecodeMs: (float) $nativeContext->t_eval_ms,
                    nativeSampleMs: (float) $nativeSampler->t_sample_ms,
                    nativeGraphReuseCount: (int) $nativeContext->n_reused,
                ),
            ];

            $this->library->logger()->info('Completed generation.', [
                'finish_reason' => $finishReason,
                'prompt_tokens' => $promptEvaluation?->tokenCount ?? 0,
                'generated_tokens' => count($generatedTokenIds),
                'decode_duration_us' => $decodeDurationUs,
                'total_duration_us' => $totalDurationUs,
                'native_decode_ms' => (float) $nativeContext->t_eval_ms,
                'native_prompt_eval_ms' => (float) $nativeContext->t_p_eval_ms,
            ]);
        } finally {
            $ffi->llama_sampler_free($sampler);
        }
    }

    private function buildSampler(GenerationConfig $config): CData
    {
        $ffi = $this->library->ffi();
        $params = $ffi->llama_sampler_chain_default_params();
        $params->no_perf = false;
        $chain = $ffi->llama_sampler_chain_init($params);

        if ($config->temperature <= 0.0) {
            $ffi->llama_sampler_chain_add($chain, $ffi->llama_sampler_init_greedy());

            return $chain;
        }

        if ($config->topK > 0) {
            $ffi->llama_sampler_chain_add($chain, $ffi->llama_sampler_init_top_k($config->topK));
        }

        if ($config->topP < 1.0) {
            $ffi->llama_sampler_chain_add($chain, $ffi->llama_sampler_init_top_p($config->topP, 1));
        }

        if ($config->minP > 0.0) {
            $ffi->llama_sampler_chain_add($chain, $ffi->llama_sampler_init_min_p($config->minP, 1));
        }

        $ffi->llama_sampler_chain_add($chain, $ffi->llama_sampler_init_temp($config->temperature));
        $ffi->llama_sampler_chain_add($chain, $ffi->llama_sampler_init_dist($config->seed));

        return $chain;
    }

    private function pieceForToken(int $tokenId, bool $unparseSpecial, ?CData &$buffer = null, int &$bufferCapacity = 256): string
    {
        $ffi = $this->library->ffi();
        $buffer ??= $this->library->new(sprintf('char[%d]', $bufferCapacity));
        $written = (int) $ffi->llama_token_to_piece(
            $this->model->vocabHandle(),
            $tokenId,
            $buffer,
            $bufferCapacity,
            0,
            $unparseSpecial,
        );

        if ($written >= 0) {
            return $this->library->string($buffer, $written);
        }

        $size = -$written + 1;
        $bufferCapacity = $size;
        $buffer = $this->library->new(sprintf('char[%d]', $bufferCapacity));
        $written = (int) $ffi->llama_token_to_piece(
            $this->model->vocabHandle(),
            $tokenId,
            $buffer,
            $bufferCapacity,
            0,
            $unparseSpecial,
        );
        if ($written < 0) {
            throw new BackendException(sprintf('Failed to decode token %d into text.', $tokenId));
        }

        return $this->library->string($buffer, $written);
    }

    /**
     * @param list<string> $stopStrings
     */
    private function endsWithStopString(string $text, array $stopStrings): bool
    {
        foreach ($stopStrings as $stopString) {
            if ($stopString !== '' && str_ends_with($text, $stopString)) {
                return true;
            }
        }

        return false;
    }

    /**
     * @param list<string> $stopStrings
     */
    private function trimStopStrings(string $text, array $stopStrings): string
    {
        foreach ($stopStrings as $stopString) {
            if ($stopString !== '' && str_ends_with($text, $stopString)) {
                return substr($text, 0, -strlen($stopString));
            }
        }

        return $text;
    }

    private function assertOpen(): void
    {
        if ($this->closed) {
            throw new BackendException('Session has already been closed.');
        }
    }
}
