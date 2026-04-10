<?php

declare(strict_types=1);

use PhpCsFixer\Config;
use PhpCsFixer\Finder;

$finder = (new Finder())
    ->files()
    ->in([
        __DIR__ . '/src',
        __DIR__ . '/tests',
        __DIR__ . '/scripts',
        __DIR__ . '/bin',
    ])
    ->name('*.php');

return (new Config())
    ->setRiskyAllowed(false)
    ->setCacheFile(__DIR__ . '/.php-cs-fixer.cache')
    ->setRules([
        '@PER-CS' => true,
        'no_unused_imports' => true,
        'ordered_imports' => ['sort_algorithm' => 'alpha'],
        'single_import_per_statement' => true,
    ])
    ->setFinder($finder);
