#!/usr/bin/env bash
# build and install liblevvy.so into the neovim config (linux counterpart
# of install.cmd)
set -euo pipefail
cd "$(dirname "$0")"

zig build -Doptimize=ReleaseFast

DEST="${XDG_CONFIG_HOME:-$HOME/.config}/nvim/zig"
mkdir -p "$DEST"
cp zig-out/lib/liblevvy.so "$DEST/"
echo "liblevvy.so installed to $DEST"
