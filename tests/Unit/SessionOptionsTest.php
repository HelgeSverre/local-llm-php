<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use InvalidArgumentException;
use HelgeSverre\LocalLlm\Backend\SessionOptions;
use PHPUnit\Framework\TestCase;

final class SessionOptionsTest extends TestCase
{
    public function testItRejectsBatchLargerThanContext(): void
    {
        $this->expectException(InvalidArgumentException::class);

        new SessionOptions(contextTokens: 128, batchSize: 256);
    }

    public function testItRejectsMicroBatchLargerThanBatch(): void
    {
        $this->expectException(InvalidArgumentException::class);

        new SessionOptions(contextTokens: 512, batchSize: 128, microBatchSize: 256);
    }
}
