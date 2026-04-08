<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\Support\EnvironmentChecker;
use HelgeSverre\LocalLlm\Support\RuntimePlatform;
use PHPUnit\Framework\TestCase;

final class EnvironmentCheckerTest extends TestCase
{
    public function testItReportsAMissingLibraryAsBlocking(): void
    {
        $report = EnvironmentChecker::inspectForLlamaCpp('/path/that/does/not/exist/' . RuntimePlatform::sharedLibraryBasename());

        self::assertTrue($report->hasBlockingIssues());
        self::assertNotEmpty($report->blockingIssues());
    }
}
