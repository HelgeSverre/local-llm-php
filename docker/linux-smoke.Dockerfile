FROM php:8.3-cli-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        $PHPIZE_DEPS \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libffi-dev \
        pkg-config \
        unzip \
    && docker-php-ext-install ffi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=composer:2 /usr/bin/composer /usr/local/bin/composer

WORKDIR /app
COPY . .

RUN ./scripts/smoke-install.sh
