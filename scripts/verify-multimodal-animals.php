<?php

declare(strict_types=1);

use HelgeSverre\LocalLlm\Chat\ChatMessage;
use HelgeSverre\LocalLlm\Generation\GenerationConfig;
use HelgeSverre\LocalLlm\Generation\MediaInput;
use HelgeSverre\LocalLlm\LocalLlm;

require dirname(__DIR__) . '/vendor/autoload.php';

$libraryPath = getenv('LOCAL_LLM_FFI_LLAMA_LIB') ?: '';
$modelPath = getenv('LOCAL_LLM_FFI_MM_MODEL') ?: '';
$multimodalProjectorPath = getenv('LOCAL_LLM_FFI_MM_MMPROJ') ?: '';
$manifestPath = getenv('LOCAL_LLM_FFI_MM_ANIMAL_MANIFEST') ?: '';

foreach ([
    'LOCAL_LLM_FFI_LLAMA_LIB' => $libraryPath,
    'LOCAL_LLM_FFI_MM_MODEL' => $modelPath,
    'LOCAL_LLM_FFI_MM_MMPROJ' => $multimodalProjectorPath,
    'LOCAL_LLM_FFI_MM_ANIMAL_MANIFEST' => $manifestPath,
] as $name => $value) {
    if (!is_string($value) || trim($value) === '') {
        fwrite(STDERR, $name . " must be set.\n");
        exit(1);
    }
}

if (!is_file($libraryPath) || !is_file($modelPath) || !is_file($multimodalProjectorPath) || !is_file($manifestPath)) {
    fwrite(STDERR, "Library, model, mmproj, and manifest paths must point to existing files.\n");
    exit(1);
}

$manifest = json_decode((string) file_get_contents($manifestPath), true, flags: JSON_THROW_ON_ERROR);
if (!is_array($manifest) || $manifest === []) {
    fwrite(STDERR, "Manifest must be a non-empty JSON array.\n");
    exit(1);
}

$manifestDirectory = dirname($manifestPath);

$runtime = LocalLlm::llamaCppRuntime(
    modelPath: $modelPath,
    libraryPath: $libraryPath,
    multimodalProjectorPath: $multimodalProjectorPath,
    gpuLayers: (int) (getenv('LOCAL_LLM_FFI_MM_GPU_LAYERS') ?: getenv('LOCAL_LLM_FFI_GPU_LAYERS') ?: '99'),
    captureNativeLogs: false,
);

try {
    foreach ($manifest as $index => $case) {
        if (!is_array($case)) {
            throw new RuntimeException(sprintf('Manifest case %d must be an object.', $index));
        }

        $label = trim((string) ($case['label'] ?? ''));
        $imagePath = trim((string) ($case['image'] ?? ''));
        $expectedPattern = trim((string) ($case['expected_pattern'] ?? ''));

        if ($label === '' || $imagePath === '' || $expectedPattern === '') {
            throw new RuntimeException(sprintf('Manifest case %d must contain label, image, and expected_pattern.', $index));
        }

        if (!str_starts_with($imagePath, DIRECTORY_SEPARATOR)) {
            $imagePath = $manifestDirectory . '/' . $imagePath;
        }

        if (!is_file($imagePath)) {
            throw new RuntimeException(sprintf('Image for case "%s" not found at "%s".', $label, $imagePath));
        }

        $chat = $runtime->newChatSession();
        try {
            $result = $chat->generate(
                [
                    ChatMessage::userWithMedia(
                        'What animal is shown in this image? Answer in a few words.',
                        [MediaInput::fromFile($imagePath)],
                    ),
                ],
                new GenerationConfig(maxTokens: 24, temperature: 0.0),
            );
        } finally {
            $chat->close();
        }

        $answer = trim($result->text);
        $matched = preg_match($expectedPattern, $answer) === 1;

        printf(
            "[%s] %s\n",
            $matched ? 'PASS' : 'FAIL',
            $label,
        );
        printf("  image: %s\n", $imagePath);
        printf("  answer: %s\n", $answer === '' ? '(empty)' : $answer);
        printf("  expected: %s\n", $expectedPattern);

        if (!$matched) {
            exit(2);
        }
    }
} finally {
    $runtime->close();
}
