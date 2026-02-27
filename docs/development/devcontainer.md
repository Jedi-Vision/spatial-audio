# Devcontainer

The repository includes a Linux devcontainer with:
- CMake + Ninja
- pkg-config
- ZeroMQ/PortAudio/GLFW system deps
- `vcpkg` bootstrap at `/home/vscode/vcpkg`

## Use
1. Open the repo in VS Code.
2. Run “Reopen in Container”.
3. After container setup:
   ```bash
   cmake --preset vcpkg
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   ```

## Troubleshooting
- If you changed `.devcontainer/devcontainer.json`, run “Dev Containers: Rebuild Container”.
- `VCPKG_ROOT` must point to a writable path for `remoteUser` (`vscode`).
