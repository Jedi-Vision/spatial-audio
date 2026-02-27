# System Overview

The project is split into executable apps and shared libraries:

- `jsa_protocol`: frozen v1 frame parsing/serialization for 2D and 3D payloads.
- `jsa_core`: shared utilities (`wav_io`, `math_coords`, `resource_locator`).
- `jsa_tracking`: live object trackers for 2D and 3D streams.

## Apps
- `jsa-demo`: offline spatial sweep demo.
- `jsa-offline-render`: JSON/JSONL detection stream to binaural WAV.
- `jsa-live-2d`: REP receiver for 2D socket payloads, speaker render output.
- `jsa-live-3d`: REP receiver for 3D payloads, tone/song render output.
- `jsa-orbit-stream`: REQ synthetic 3D stream generator.
- `jsa-visual-monitor`: REP visualizer and optional REQ forwarder.

## Asset Resolution
All apps resolve runtime assets with:
1. `--assets-root <path>`
2. `JSA_ASSET_ROOT`
3. repo-local `assets/`
