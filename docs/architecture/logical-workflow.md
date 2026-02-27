# Logical Workflow

## Offline Render Path
1. `jsa-offline-render` loads detection frames and source audio.
2. Frames are tracked/interpolated into object positions.
3. Steam Audio direct + binaural effects are rendered per frame.
4. Final stereo result is written to output WAV.

## Live Stream Path
1. Producer (`jsa-orbit-stream` or external CV pipeline) sends REQ frame payloads.
2. Receiver (`jsa-visual-monitor` and/or `jsa-live-2d` / `jsa-live-3d`) parses payloads.
3. Receiver returns ACK (`'0'` success, `'1'` parse-failure) over REP.
4. Live renderers track objects and emit continuous audio.

## Forwarded Monitoring Path
1. `jsa-visual-monitor` binds upstream REP endpoint.
2. Parsed payloads are optionally forwarded to downstream REQ endpoint.
3. Downstream ACK is required before forward success is counted.
4. Retry/reconnect behavior is controlled via forward timeout/retry flags.
