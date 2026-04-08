<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Tests\Unit;

use HelgeSverre\LocalLlm\Support\AppleSiliconTier;
use HelgeSverre\LocalLlm\Support\SupportedModelCatalog;
use PHPUnit\Framework\TestCase;

final class SupportedModelCatalogTest extends TestCase
{
    public function testItReturnsKnownProfiles(): void
    {
        $profile = SupportedModelCatalog::get('granite3.3-2b-ollama');

        self::assertSame('granite3.3:2b', $profile->ollamaModelName);
        self::assertTrue($profile->ollamaManaged);
        self::assertTrue($profile->supportsTier(AppleSiliconTier::GB16));
        self::assertSame(4096, $profile->recommendedSessionOptions(AppleSiliconTier::GB16)->contextTokens);
    }

    public function testItRejectsUnsupportedTiers(): void
    {
        $profile = SupportedModelCatalog::get('qwen2.5-coder-14b-instruct-q5-k-m');

        $this->expectException(\InvalidArgumentException::class);
        $profile->recommendedSessionOptions(AppleSiliconTier::GB32);
    }

    public function testLocalLlmFacadeReturnsQualifiedProfiles(): void
    {
        $profiles = \HelgeSverre\LocalLlm\LocalLlm::supportedModels();

        self::assertNotEmpty($profiles);
        self::assertContainsOnlyInstancesOf(\HelgeSverre\LocalLlm\Support\SupportedModelProfile::class, $profiles);
    }
}
