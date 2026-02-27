# Jedi Spatial Audio

A C++ project for spatial audio processing using Steam Audio, designed to spatialize audio sources based on object detection data from computer vision systems (e.g., YOLO).

## Features

- **3D Spatial Audio**: Uses Steam Audio's binaural rendering to create immersive 3D audio experiences
- **Object Tracking**: Processes object detection data (2D position + depth) and converts to 3D spatial audio
- **Multi-Source Support**: Handles up to 10 simultaneous audio sources with independent spatialization
- **Smooth Interpolation**: Interpolates object positions between video frames for continuous audio playback
- **Custom HRTF**: Supports custom SOFA HRTF files for personalized spatial audio

## Requirements

- CMake 4.1 or higher
- C++ compiler with C++17 support
- [vcpkg](https://vcpkg.io/) for dependency management
- Steam Audio SDK (installed via vcpkg)

## Dependencies

The project uses the following dependencies (managed via vcpkg):

- **steam-audio**: Steam Audio SDK for spatial audio processing
- **fmt**: Fast formatting library
- **nlohmann-json**: JSON parsing library
- **raylib**: Realtime 2D/3D rendering for the live visual monitor

## Building

1. **Install vcpkg** (if not already installed):
   ```bash
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.sh  # On macOS/Linux
   ```

2. **Configure CMake with vcpkg**:
   ```bash
   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
   ```

3. **Build the project**:
   ```bash
   cmake --build build
   ```

The executables will be in the `build/` directory:
- `build/spatial-audio-demo` - Basic spatial audio demo
- `build/object-spatial-audio` - Object tracking spatial audio
- `build/socket-spatial-audio-live` - Live socket receiver + speaker playback
- `build/spatial_audio_live_new` - 3D socket receiver + speaker spatial audio
- `build/socket-orbit-stream-3d` - Synthetic 3D orbit stream generator
- `build/render-visual-stream` - Realtime 3D socket visualizer + forwarder

## Usage

### Spatial Audio Demo

Basic demo that spatializes a single audio source moving in 3D space:

```bash
./build/spatial-audio-demo [OPTIONS]
```

Options:
- `--hrtf <default|custom>` - Use default or custom HRTF (default: custom)
- `--export-trajectory, -e` - Export trajectory data to CSV file
- `--help, -h` - Show help message

Input: `beep_1.wav` (must be in project root)
Output: `output_binaural.wav`

### Object Spatial Audio

Processes object detection data and generates spatialized audio for multiple objects:

```bash
./build/object-spatial-audio [INPUT_JSON] [OUTPUT_WAV] [OPTIONS]
```

Options:
- `--hrtf <default|custom>` - Use default or custom HRTF (default: custom)
- `--help, -h` - Show help message

**Input JSON Format** (JSONL - one JSON object per line):
```json
{"frame": 0, "timestamp_ms": 0.0, "objects": [{"id": 1, "x": 0.5, "y": 0.5, "depth": 2.0}]}
{"frame": 1, "timestamp_ms": 33.33, "objects": [{"id": 1, "x": 0.52, "y": 0.51, "depth": 2.0}, {"id": 2, "x": 0.3, "y": 0.7, "depth": 3.0}]}
```

Fields:
- `frame`: Frame number (integer)
- `timestamp_ms`: Timestamp in milliseconds (float)
- `objects`: Array of detected objects
  - `id`: Unique object identifier (integer)
  - `x`: Normalized X position [0, 1] (float)
  - `y`: Normalized Y position [0, 1] (float)
  - `depth`: Depth in meters (float)
  - `confidence`: Optional confidence score (float)
  - `class`: Optional class name (string)

**Audio File Mapping**:
- Object ID 1 uses `beep_1.wav`
- Object ID 2 uses `beep_2.wav`

**Example**:
```bash
./build/object-spatial-audio sample_detections.json output.wav
```

### Live Socket-to-Speaker Spatial Audio

Receives real-time object stream messages over ZeroMQ IPC, spatializes each frame with Steam Audio, and plays directly to your default laptop speaker through PortAudio.

```bash
./build/socket-spatial-audio-live [OPTIONS]
```

Options:
- `--ipc <endpoint>` - ZeroMQ endpoint (default: `ipc:///tmp/jv/audio/0.sock`)
- `--audio <wav>` - Source beep/sample WAV (default: `beep_1.wav`)
- `--hrtf <default|custom>` - HRTF mode (default: `default`)
- `--device-index <int>` - PortAudio output device index (default: `-1`, system default)
- `--help, -h` - Show help message

Example:
```bash
./build/socket-spatial-audio-live --ipc ipc:///tmp/jv/audio/0.sock --audio beep_1.wav --hrtf default
```

For integration with `jedi-vision-nano-code`, run the vision pipeline with:
- `output_to: socket`
- `serial_type: struct`

### 3D Orbit Stream Generator

Generates synthetic 3D object positions and streams them over ZeroMQ IPC for live spatial-audio testing.

```bash
./build/socket-orbit-stream-3d [OPTIONS]
```

Options:
- `--ipc <endpoint>` - ZeroMQ endpoint (default: `ipc:///tmp/jv/audio/0.sock`)
- `--fps <value>` - Frames per second (default: `30.0`)
- `--radius <meters>` - Base orbit radius in meters (default: `2.0`)
- `--period-sec <sec>` - Angular orbit period in seconds (default: `8.0`)
- `--motion-mode <orbit|wavy|single-wavy>` - Motion profile (default: `orbit`)
- `--radial-amp <meters>` - Radius modulation amplitude for `wavy` mode (default: `0.75`)
- `--radial-period-sec <sec>` - Radius modulation period for `wavy` mode (default: `5.0`)
- `--phase-offset-deg <deg>` - Object 2 radial phase offset for `wavy` mode (default: `180.0`)
- `--y <meters>` - Constant Y height (default: `0.0`)
- `--id1 <int>`, `--id2 <int>` - Object IDs (defaults: `1`, `2`)
- `--label1 <int>`, `--label2 <int>` - Object labels (defaults: `0`, `0`)

Orbit mode example:
```bash
./build/socket-orbit-stream-3d --ipc ipc:///tmp/jv/audio/0.sock --motion-mode orbit --radius 2.0 --period-sec 8.0
```

Wavy mode example (changes distance + position over time):
```bash
./build/socket-orbit-stream-3d --ipc ipc:///tmp/jv/audio/0.sock --motion-mode wavy --radius 2.0 --period-sec 8.0 --radial-amp 0.75 --radial-period-sec 5.0 --phase-offset-deg 180
```

Single-wavy mode example (single moving object with changing distance):
```bash
./build/socket-orbit-stream-3d --ipc ipc:///tmp/jv/audio/0.sock --motion-mode single-wavy --radius 2.0 --period-sec 8.0 --radial-amp 0.75 --radial-period-sec 5.0 --id1 1 --label1 0
```

### Realtime 3D Visual Monitor

Receives orbit-stream frames over ZeroMQ IPC, ACKs upstream, renders objects in a 3D window, and optionally forwards payloads to another endpoint so audio can run simultaneously.

```bash
./build/render-visual-stream [OPTIONS]
```

Options:
- `--ipc <endpoint>` - Upstream endpoint to bind as REP (default: `ipc:///tmp/jv/audio/0.sock`)
- `--forward-ipc <endpoint>` - Downstream endpoint to forward as REQ (default: `ipc:///tmp/jv/audio/1.sock`)
- `--no-forward` - Disable forwarding and run as visualization-only receiver
- `--forward-send-timeout-ms <ms>` - Forward send timeout (default: `100`)
- `--forward-recv-timeout-ms <ms>` - Forward ACK timeout (default: `100`)
- `--forward-retries <count>` - Forward retry count (default: `1`)
- `--width <pixels>` - Window width (default: `1280`)
- `--height <pixels>` - Window height (default: `720`)
- `--fps <value>` - Render target FPS (default: `60`)
- `--trail-seconds <value>` - Trail retention window in seconds (default: `2.0`)
- `--help, -h` - Show help message

Visualization-only example:
```bash
./build/render-visual-stream --ipc ipc:///tmp/jv/audio/0.sock --no-forward
```

Forwarding example:
```bash
./build/render-visual-stream --ipc ipc:///tmp/jv/audio/0.sock --forward-ipc ipc:///tmp/jv/audio/1.sock --forward-send-timeout-ms 100 --forward-recv-timeout-ms 100 --forward-retries 1
```

Live test trio:
Terminal A (audio receiver on forwarded endpoint):
```bash
./build/spatial_audio_live_new --ipc ipc:///tmp/jv/audio/1.sock --hrtf default --source-mode tones
```
Songs mode example (moving songs instead of tones):
```bash
./build/spatial_audio_live_new --ipc ipc:///tmp/jv/audio/1.sock --hrtf default --source-mode songs --song-a lucky.wav --song-b september.wav
```
`spatial_audio_live_new` supports two source modes:
- `tones` (default): stable C-major pentatonic-bell tones per object ID.
- `songs`: continuous per-object song playback with deterministic assignment (`hash(object_id) % 2`) between `song-a` and `song-b`.
In `songs` mode, each object keeps its own playback cursor while active and the song restarts from the beginning when that object disappears and later reappears.

Terminal B (visualizer + forwarder):
```bash
./build/render-visual-stream --ipc ipc:///tmp/jv/audio/0.sock --forward-ipc ipc:///tmp/jv/audio/1.sock --forward-send-timeout-ms 100 --forward-recv-timeout-ms 100 --forward-retries 1
```

Terminal C (orbit generator):
```bash
./build/socket-orbit-stream-3d --ipc ipc:///tmp/jv/audio/0.sock --motion-mode wavy --radius 2.0 --period-sec 8.0 --radial-amp 0.75 --radial-period-sec 5.0 --phase-offset-deg 180
```

## Project Structure

```
jedi-spatial-audio/
├── CMakeLists.txt              # CMake build configuration
├── vcpkg.json                  # vcpkg dependencies
├── spatial-audio-demo.cpp      # Basic spatial audio demo
├── object-spatial-audio.cpp    # Object tracking spatial audio
├── socket-spatial-audio-live.cpp # Live socket receiver + speaker playback
├── spatial_audio_live_new.cpp  # 3D socket receiver + speaker playback
├── socket_orbit_stream_3d.cpp  # Synthetic 3D orbit stream generator
├── render_visual_stream.cpp    # Realtime 3D visual monitor + forwarder
├── socket_payload_parser.h     # Struct socket payload parser API
├── socket_payload_parser.cpp   # Struct socket payload parser implementation
├── sample_detections.json      # Sample object detection data
├── D2_HRIR_SOFA/               # HRTF SOFA files
│   ├── D2_44K_16bit_256tap_FIR_SOFA.sofa
│   ├── D2_48K_24bit_256tap_FIR_SOFA.sofa
│   └── D2_96K_24bit_512tap_FIR_SOFA.sofa
└── build/                      # Build directory (generated)
```

## Technical Details

### Coordinate System

The system uses Steam Audio's right-handed coordinate system:
- **+X**: Right
- **+Y**: Up
- **+Z**: Backward (opposite of forward)

Object detection coordinates (2D + depth) are converted to 3D positions using:
- Camera FOV: 60° horizontal, 45° vertical (configurable)
- Normalized coordinates [0, 1] where (0, 0) is top-left

### Audio Processing

- **Sample Rate**: 44.1 kHz
- **Frame Size**: 1024 samples
- **Video Frame Rate**: 30 FPS
- **Max Simultaneous Objects**: 10

### Spatialization Features

- **Binaural Rendering**: Uses HRTF (Head-Related Transfer Function) for 3D audio
- **Distance Attenuation**: Inverse distance model for realistic volume falloff
- **Smooth Interpolation**: Linear interpolation between detection frames
- **Fade In/Out**: Objects fade in when appearing and fade out when disappearing

## Integration with Object Detection

This project is designed to work with object detection pipelines (e.g., YOLO). The typical workflow:

1. **Object Detection**: Run YOLO or similar on video frames (30 FPS)
2. **Depth Estimation**: Get depth for each detected object
3. **Export Data**: Convert detection results to JSON format
4. **Spatial Audio**: Process JSON with `object-spatial-audio` to generate spatialized audio

## License

[Add your license here]

## Acknowledgments

- [Steam Audio](https://valvesoftware.github.io/steam-audio/) by Valve Software
- [vcpkg](https://vcpkg.io/) by Microsoft
