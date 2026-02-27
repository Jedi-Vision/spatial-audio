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

## Runtime Resolution Order
Applications resolve asset paths in this order:
1. `--assets-root <path>`
2. `JSA_ASSET_ROOT`
3. repo-local `assets/`
