<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\Support\Statistics;
use PHPUnit\Framework\TestCase;

final class StatisticsTest extends TestCase
{
    public function testItSummarizesNumericSamples(): void
    {
        $summary = Statistics::summarize([10.0, 20.0, 30.0, 40.0]);

        self::assertSame(4, $summary['count']);
        self::assertSame(10.0, $summary['min']);
        self::assertSame(40.0, $summary['max']);
        self::assertSame(25.0, $summary['mean']);
        self::assertSame(25.0, $summary['median']);
        self::assertGreaterThan(0.0, $summary['stddev']);
    }
}
