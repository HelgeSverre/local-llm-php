<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Backend;

interface BackendInterface
{
    public function name(): string;

    public function loadModel(ModelOptions $options): ModelInterface;
}
