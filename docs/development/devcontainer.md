# Devcontainer

The repository includes a Linux devcontainer with:
- CMake + Ninja
- pkg-config
- ZeroMQ/PortAudio/GLFW system deps
- `vcpkg` bootstrap

## Use
1. Open the repo in VS Code.
2. Run “Reopen in Container”.
3. After container setup:
   ```bash
   cmake --preset vcpkg
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   ```
