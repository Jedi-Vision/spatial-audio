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

When `PULSE_SERVER` is set, `jsa-live-3d` and `jsa-live-2d` now prefer a Pulse-backed
PortAudio output device before falling back to PortAudio's default output device.

Optional playback stability tuning:
```bash
./build/jsa-live-3d \
  --ipc ipc:///tmp/jv/audio/1.sock \
  --source-mode songs \
  --output-latency-mode high \
  --audio-buffer-ms 60 \
  --stream-timeout-ms 34
```
`--stream-timeout-ms` controls only ZeroMQ receive timeout. Audio pacing is callback-driven.

Terminal B:
```bash
./build/jsa-visual-monitor --ipc ipc:///tmp/jv/audio/0.sock --forward-ipc ipc:///tmp/jv/audio/1.sock
```

Terminal C:
```bash
./build/jsa-orbit-stream --ipc ipc:///tmp/jv/audio/0.sock --motion-mode wavy
```

## Docker On Jetson
To route container audio through the host Jetson PulseAudio sink instead of an HDMI-biased
PortAudio default device, make sure the container runtime provides:

- `PULSE_SERVER=unix:/run/user/$UID/pulse/native`
- `XDG_RUNTIME_DIR=/run/user/$UID`
- a bind mount for `/run/user/$UID/pulse`
- a bind mount for `/etc/machine-id`
- access to `/dev/snd`
- `--group-add audio`

The repo's `start_container.sh` already exports and mounts these PulseAudio runtime bits
when the host Pulse socket exists, so `jsa-live-3d` inside that container should auto-pick
the Pulse output path without needing `--device-index`.
