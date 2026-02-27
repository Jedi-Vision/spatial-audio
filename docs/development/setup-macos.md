# macOS Setup

## Prerequisites
- Xcode command line tools
- Homebrew
- `vcpkg` clone

## Bootstrap
```bash
./scripts/bootstrap_macos.sh
```

## Environment
```bash
export VCPKG_ROOT=/absolute/path/to/vcpkg
```

## Configure + Build + Test
```bash
cmake --preset vcpkg
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Assets
```bash
./scripts/fetch_assets.sh
```
