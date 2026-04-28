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

- `PULSE_SERVER=unix:/tmp/pulse/native`
- a bind mount from `/run/user/1000/pulse/native` to `/tmp/pulse/native`
- a read-only bind mount from `$HOME/.config/pulse/cookie` to `/root/.config/pulse/cookie`
- a bind mount for `/etc/machine-id`

The image also configures `/etc/asound.conf` so ALSA's default device routes through
PulseAudio. The repo's `start_container.sh` provides these PulseAudio runtime bits, so
`jsa-live-3d` and `jsa-live-2d` should auto-pick the Pulse output path without needing
`--device-index`.
