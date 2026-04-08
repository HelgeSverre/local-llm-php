<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Support;

use HelgeSverre\LocalLlm\Backend\BackendException;

final readonly class EnvironmentReport
{
    /**
     * @param list<EnvironmentIssue> $issues
     */
    public function __construct(
        public string $libraryPath,
        public array $issues,
    ) {
    }

    public function hasBlockingIssues(): bool
    {
        foreach ($this->issues as $issue) {
            if ($issue->isBlocking()) {
                return true;
            }
        }

        return false;
    }

    /**
     * @return list<EnvironmentIssue>
     */
    public function blockingIssues(): array
    {
        return array_values(array_filter(
            $this->issues,
            static fn (EnvironmentIssue $issue): bool => $issue->isBlocking(),
        ));
    }

    /**
     * @return list<EnvironmentIssue>
     */
    public function warnings(): array
    {
        return array_values(array_filter(
            $this->issues,
            static fn (EnvironmentIssue $issue): bool => !$issue->isBlocking(),
        ));
    }

    public function assertReady(): void
    {
        if (!$this->hasBlockingIssues()) {
            return;
        }

        $messages = array_map(
            static fn (EnvironmentIssue $issue): string => $issue->message,
            $this->blockingIssues(),
        );

        throw new BackendException('Environment is not ready for llama.cpp FFI: ' . implode(' ', $messages));
    }
}
