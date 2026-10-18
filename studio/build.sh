#!/usr/bin/env sh
# Build the browser view into web/pkg. Serve with:
#   WEB_ROOT=studio/web node server.mjs     (from the repo root)
set -e
cd "$(dirname "$0")"
cargo build --release --lib --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir web/pkg --out-name studio \
  target/wasm32-unknown-unknown/release/studio.wasm
