<?php

declare(strict_types=1);

return [
    [
        'name' => 'short',
        'prompt' => 'Write exactly one sentence about why local inference can be attractive on Apple Silicon.',
        'max_tokens' => 64,
    ],
    [
        'name' => 'long',
        'prompt' => trim(str_repeat(
            "You are evaluating a local LLM runtime for a PHP package. Summarize the trade-offs between prompt throughput, decode throughput, warmup cost, context reuse, and streaming latency. Keep the analysis concrete and concise. ",
            32,
        )),
        'max_tokens' => 96,
    ],
];
