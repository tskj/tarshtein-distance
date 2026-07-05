#!/usr/bin/env bash
# build and install the levvy shared library into the neovim config
# (linux/macos counterpart of install.cmd). the nvim config loads it
# automatically -- see lua/user/levvy.lua.
set -euo pipefail
cd "$(dirname "$0")"

zig build -Doptimize=ReleaseFast

DEST="${XDG_CONFIG_HOME:-$HOME/.config}/nvim/zig"
mkdir -p "$DEST"

# zig emits liblevvy.so on linux, liblevvy.dylib on macos
copied=0
for lib in zig-out/lib/liblevvy.so zig-out/lib/liblevvy.dylib; do
  if [ -f "$lib" ]; then
    cp "$lib" "$DEST/"
    echo "installed $(basename "$lib") to $DEST"
    copied=1
  fi
done

if [ "$copied" -eq 0 ]; then
  echo "error: no liblevvy.{so,dylib} found in zig-out/lib" >&2
  exit 1
fi
