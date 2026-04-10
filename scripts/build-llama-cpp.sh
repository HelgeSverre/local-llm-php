#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NATIVE_DIR="$ROOT_DIR/var/native"
SOURCE_DIR="$NATIVE_DIR/llama.cpp"
BUILD_DIR="$SOURCE_DIR/build"
ARCHIVE_URL="${LLAMA_CPP_ARCHIVE_URL:-https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz}"
REFRESH="${1:-}"

mkdir -p "$NATIVE_DIR"

if [[ "$REFRESH" == "--refresh" || ! -d "$SOURCE_DIR" ]]; then
  archive_file="$(mktemp "${TMPDIR:-/tmp}/llama.cpp.XXXXXX.tar.gz")"
  rm -rf "$SOURCE_DIR"
  curl --http1.1 -L --fail --retry 5 --retry-delay 2 --retry-all-errors "$ARCHIVE_URL" -o "$archive_file"
  extracted_dir="$(
    tar -tzf "$archive_file" \
      | cut -d/ -f1 \
      | sed '/^$/d' \
      | sort -u \
      | sed -n '1p'
  )"
  rm -rf "$NATIVE_DIR/$extracted_dir"
  tar -xzf "$archive_file" -C "$NATIVE_DIR"
  mv "$NATIVE_DIR/$extracted_dir" "$SOURCE_DIR"
  rm -f "$archive_file"
fi

OS_FAMILY="$(uname -s)"
MACHINE_ARCH="$(uname -m)"
METAL_FLAG="OFF"
LIBRARY_NAME="libllama.so"
RPATH_VALUE="\$ORIGIN"

case "$OS_FAMILY" in
  Darwin)
    METAL_FLAG="ON"
    LIBRARY_NAME="libllama.dylib"
    RPATH_VALUE="@loader_path"
    ;;
  Linux)
    ;;
  *)
    echo "Unsupported host OS: $OS_FAMILY" >&2
    exit 1
    ;;
esac

CPU_COUNT="$(
  getconf _NPROCESSORS_ONLN 2>/dev/null \
    || nproc 2>/dev/null \
    || sysctl -n hw.logicalcpu 2>/dev/null \
    || echo 4
)"

extra_args=()
if [[ -n "${LLAMA_CPP_CMAKE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($LLAMA_CPP_CMAKE_ARGS)
fi

if [[ "$OS_FAMILY" == "Linux" && "$MACHINE_ARCH" =~ ^(aarch64|arm64)$ ]]; then
  extra_args+=(
    -DGGML_NATIVE=OFF
    -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16
  )
fi

cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=ON
  -DCMAKE_BUILD_RPATH="$RPATH_VALUE"
  -DCMAKE_INSTALL_RPATH="$RPATH_VALUE"
  -DGGML_METAL="$METAL_FLAG"
  -DLLAMA_BUILD_COMMON=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
)

if ((${#extra_args[@]} > 0)); then
  cmake_args+=("${extra_args[@]}")
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" "${cmake_args[@]}"

cmake --build "$BUILD_DIR" --config Release -j "$CPU_COUNT"

echo "Built llama.cpp shared library at: $BUILD_DIR/bin/$LIBRARY_NAME"
