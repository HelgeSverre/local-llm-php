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

RUN composer install --no-interaction --prefer-dist \
    && ./scripts/build-llama-cpp.sh --refresh

CMD ["php", "-d", "ffi.enable=1", "./bin/doctor"]
