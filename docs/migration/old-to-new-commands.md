# Command Migration (Legacy -> New)

Legacy command names still work as compatibility aliases in this release.

| Legacy Binary | New Binary |
| --- | --- |
| `spatial-audio-demo` | `jsa-demo` |
| `object-spatial-audio` | `jsa-offline-render` |
| `socket-spatial-audio-live` | `jsa-live-2d` |
| `spatial_audio_live_new` | `jsa-live-3d` |
| `socket-orbit-stream-3d` | `jsa-orbit-stream` |
| `render-visual-stream` | `jsa-visual-monitor` |

## Example

Old:
```bash
./build/socket-orbit-stream-3d --help
```

New:
```bash
./build/jsa-orbit-stream --help
```
