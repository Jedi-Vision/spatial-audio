# Asset Management

Large runtime assets are fetched from release bundles and verified with checksums.

## Files
- Lockfile: `assets/manifest/assets.lock.json`
- Fetch script: `scripts/fetch_assets.sh`

## Fetch
```bash
./scripts/fetch_assets.sh
```

By default the script downloads into this `assets/` directory and verifies SHA-256 for every entry in the lockfile.

## Canonical Release
- GitHub repo: `Jedi-Vision/spatial-audio`
- Release tag: `assets-v1`
- Current managed files:
  - `lucky.wav`
  - `september.wav`
  - `D2_HRIR_SOFA/D2_44K_16bit_256tap_FIR_SOFA.sofa`
  - `D2_HRIR_SOFA/D2_48K_24bit_256tap_FIR_SOFA.sofa`
  - `D2_HRIR_SOFA/D2_96K_24bit_512tap_FIR_SOFA.sofa`

## Maintainer Workflow
1. Place/update local files under `assets/`.
2. Update `assets/manifest/assets.lock.json` with exact URL + SHA-256.
3. Run:
   ```bash
   ./scripts/validate_assets_manifest.sh
   ```
4. Run:
   ```bash
   ./scripts/fetch_assets.sh
   ```
5. If replacing assets incompatibly, create a new release tag (`assets-v2`, etc.) and update `release` + URLs in the lockfile.
6. Do not overwrite existing release assets in-place unless you intentionally want all historical checkouts of the same tag to change.

## Common Failures
- `HTTP 404` during validation/fetch:
  - Release tag or asset filename does not exist in `Jedi-Vision/spatial-audio`.
- `Lockfile URL mismatch`:
  - URL does not match `https://github.com/Jedi-Vision/spatial-audio/releases/download/<release>/<filename>`.
- `Checksum mismatch`:
  - Local or downloaded file contents differ from lockfile SHA-256; recompute and update lockfile only if the content change is intended.

## Runtime Resolution Order
Applications resolve asset paths in this order:
1. `--assets-root <path>`
2. `JSA_ASSET_ROOT`
3. repo-local `assets/`
