# Build And Run

## Build
```bash
cmake --preset vcpkg
cmake --build build -j
```

## Smoke
```bash
ctest --test-dir build --output-on-failure
```

## Common Commands
```bash
./build/jsa-demo --help
./build/jsa-offline-render --help
./build/jsa-live-2d --help
./build/jsa-live-3d --help
./build/jsa-orbit-stream --help
./build/jsa-visual-monitor --help
```

## Live Trio
Terminal A:
```bash
./build/jsa-live-3d --ipc ipc:///tmp/jv/audio/1.sock --hrtf default --source-mode songs
```

Terminal B:
```bash
./build/jsa-visual-monitor --ipc ipc:///tmp/jv/audio/0.sock --forward-ipc ipc:///tmp/jv/audio/1.sock
```

Terminal C:
```bash
./build/jsa-orbit-stream --ipc ipc:///tmp/jv/audio/0.sock --motion-mode wavy
```
