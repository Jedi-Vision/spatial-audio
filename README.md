# Jedi Spatial Audio

Spatial audio tooling for offline rendering and live socket-driven audio playback.

## Prerequisites
- CMake >= 4.1

## Verify CMake
```bash
cmake --version
```

## Quick Start
```bash
./scripts/bootstrap_macos.sh   # or ./scripts/bootstrap_linux.sh
export VCPKG_ROOT=/absolute/path/to/vcpkg
cmake --preset vcpkg
cmake --build build -j
ctest --test-dir build --output-on-failure
./scripts/fetch_assets.sh
```

## Binaries
- `./build/jsa-demo`
- `./build/jsa-offline-render`
- `./build/jsa-live-2d`
- `./build/jsa-live-3d`
- `./build/jsa-orbit-stream`
- `./build/jsa-visual-monitor`

Legacy names remain available as compatibility aliases for this release.

## Documentation
- Architecture:
  - `docs/architecture/system-overview.md`
  - `docs/architecture/logical-workflow.md`
  - `docs/architecture/socket-protocol-v1.md`
- Development:
  - `docs/development/setup-macos.md`
  - `docs/development/setup-linux.md`
  - `docs/development/build-and-run.md`
  - `docs/development/devcontainer.md`
- Migration:
  - `docs/migration/old-to-new-commands.md`
- Assets:
  - `assets/README.md`
