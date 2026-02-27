# Linux Setup

## Prerequisites
- apt-based distro (Ubuntu/Debian)
- sudo access
- `vcpkg` clone
- CMake >= 4.1

## Verify CMake
```bash
cmake --version
```

## Bootstrap
```bash
./scripts/bootstrap_linux.sh
```
The bootstrap script validates that CMake is at least 4.1.0 and exits with guidance if your version is too old.

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
