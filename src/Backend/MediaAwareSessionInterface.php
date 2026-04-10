<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\Backend;

interface MediaAwareSessionInterface extends SessionInterface
{
    public function mediaMarker(): ?string;
}
