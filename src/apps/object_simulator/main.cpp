#include <raylib.h>
#include <zmq.h>

#include <jsa/protocol/frame_3d_v1.hpp>
#include <jsa/protocol/frame_serializer.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double DEFAULT_FPS = 30.0;
constexpr int DEFAULT_ID = 1;
constexpr int DEFAULT_LABEL = 0;
constexpr int DEFAULT_WIDTH = 1280;
constexpr int DEFAULT_HEIGHT = 720;
constexpr int SOCKET_TIMEOUT_MS = 500;
constexpr float BASE_STEP_METERS = 0.1f;
constexpr float FAST_STEP_METERS = 0.5f;
constexpr float SLOW_STEP_METERS = 0.02f;
constexpr double ANGLE_PI = 3.14159265358979323846;

std::atomic<bool> gRunning{true};

void handleSignal(int) {
    gRunning.store(false);
}

struct CLIOptions {
    std::string ipcEndpoint;
    bool ipcProvided = false;
    double fps = DEFAULT_FPS;
    int id = DEFAULT_ID;
    int label = DEFAULT_LABEL;
    Vector3 initialHeadPosition{0.0f, 0.0f, -1.0f};
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    bool showHelp = false;
    bool valid = true;
};

struct Stats {
    uint64_t sent = 0;
    uint64_t ackedOk = 0;
    uint64_t ackedFail = 0;
    uint64_t timeouts = 0;
    uint64_t otherErrors = 0;
    uint64_t retries = 0;
};

enum class SendStatus {
    Ok,
    ParseFailAck,
    Timeout,
    Error,
    UnknownAck,
};

struct SendResult {
    SendStatus status = SendStatus::Error;
    int errnum = 0;
    char ack = '\0';
    bool retried = false;
};

struct ObjectStats {
    double distanceMeters = 0.0;
    double horizontalFovDeg = 0.0;
    double verticalFovDeg = 0.0;
    double offCenterDeg = 0.0;
};

void printUsage(const char* executableName) {
    std::cout << "Usage: " << executableName << " --ipc <endpoint> [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --ipc <endpoint>  Required ZeroMQ endpoint\n";
    std::cout << "  --fps <value>     Frames per second (default: 30)\n";
    std::cout << "  --id <int>        Object ID (default: 1)\n";
    std::cout << "  --label <int>     Object label (default: 0)\n";
    std::cout << "  --x <meters>      Initial head-space X, +right (default: 0)\n";
    std::cout << "  --y <meters>      Initial head-space Y, +up (default: 0)\n";
    std::cout << "  --z <meters>      Initial head-space Z, -forward (default: -1)\n";
    std::cout << "  --width <pixels>  Window width (default: 1280)\n";
    std::cout << "  --height <pixels> Window height (default: 720)\n";
    std::cout << "  --help, -h        Show this help message\n";
}

bool parseDouble(const char* text, double& out) {
    if (text == nullptr) {
        return false;
    }
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) {
        return false;
    }
    out = value;
    return true;
}

bool parseInt(const char* text, int& out) {
    if (text == nullptr) {
        return false;
    }
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < INT32_MIN || value > INT32_MAX) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

CLIOptions parseCommandLine(int argc, char* argv[]) {
    CLIOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            options.showHelp = true;
            return options;
        }

        auto requireValue = [&](const char* optionName) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << optionName << "\n";
                options.valid = false;
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--ipc") {
            const char* value = requireValue("--ipc");
            if (value != nullptr) {
                options.ipcEndpoint = value;
                options.ipcProvided = true;
            }
        } else if (arg == "--fps") {
            double value = 0.0;
            const char* text = requireValue("--fps");
            if (text != nullptr && parseDouble(text, value)) {
                options.fps = value;
            } else if (text != nullptr) {
                std::cerr << "Invalid --fps value: " << text << "\n";
                options.valid = false;
            }
        } else if (arg == "--id") {
            const char* text = requireValue("--id");
            if (text != nullptr && !parseInt(text, options.id)) {
                std::cerr << "Invalid --id value: " << text << "\n";
                options.valid = false;
            }
        } else if (arg == "--label") {
            const char* text = requireValue("--label");
            if (text != nullptr && !parseInt(text, options.label)) {
                std::cerr << "Invalid --label value: " << text << "\n";
                options.valid = false;
            }
        } else if (arg == "--x" || arg == "--y" || arg == "--z") {
            double value = 0.0;
            const char* text = requireValue(arg.c_str());
            if (text == nullptr || !parseDouble(text, value)) {
                if (text != nullptr) {
                    std::cerr << "Invalid " << arg << " value: " << text << "\n";
                }
                options.valid = false;
            } else if (arg == "--x") {
                options.initialHeadPosition.x = static_cast<float>(value);
            } else if (arg == "--y") {
                options.initialHeadPosition.y = static_cast<float>(value);
            } else {
                options.initialHeadPosition.z = static_cast<float>(value);
            }
        } else if (arg == "--width" || arg == "--height") {
            int value = 0;
            const char* text = requireValue(arg.c_str());
            if (text == nullptr || !parseInt(text, value)) {
                if (text != nullptr) {
                    std::cerr << "Invalid " << arg << " value: " << text << "\n";
                }
                options.valid = false;
            } else if (arg == "--width") {
                options.width = value;
            } else {
                options.height = value;
            }
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            options.valid = false;
        }
    }

    if (!options.ipcProvided) {
        std::cerr << "--ipc is required\n";
        options.valid = false;
    }
    if (options.fps <= 0.0) {
        std::cerr << "--fps must be > 0\n";
        options.valid = false;
    }
    if (options.width <= 0 || options.height <= 0) {
        std::cerr << "--width and --height must be > 0\n";
        options.valid = false;
    }

    return options;
}

void closeSocket(void*& socket) {
    if (socket != nullptr) {
        zmq_close(socket);
        socket = nullptr;
    }
}

void applySocketOption(void* socket, int option, int value, const char* name) {
    if (zmq_setsockopt(socket, option, &value, sizeof(value)) != 0) {
        std::cerr << "Warning: failed to set " << name << ": " << zmq_strerror(zmq_errno()) << "\n";
    }
}

void* createReqSocket(void* context, const std::string& endpoint) {
    void* socket = zmq_socket(context, ZMQ_REQ);
    if (socket == nullptr) {
        std::cerr << "Error: failed to create ZMQ_REQ socket: " << zmq_strerror(zmq_errno()) << "\n";
        return nullptr;
    }

    applySocketOption(socket, ZMQ_LINGER, 0, "ZMQ_LINGER");
    applySocketOption(socket, ZMQ_SNDTIMEO, SOCKET_TIMEOUT_MS, "ZMQ_SNDTIMEO");
    applySocketOption(socket, ZMQ_RCVTIMEO, SOCKET_TIMEOUT_MS, "ZMQ_RCVTIMEO");

    if (zmq_connect(socket, endpoint.c_str()) != 0) {
        std::cerr << "Error: failed to connect " << endpoint << ": " << zmq_strerror(zmq_errno()) << "\n";
        zmq_close(socket);
        return nullptr;
    }

    return socket;
}

SendResult sendFrameWithSingleRetry(void*& socket,
                                    void* context,
                                    const std::string& endpoint,
                                    const std::vector<uint8_t>& payload) {
    SendResult result;
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (socket == nullptr) {
            socket = createReqSocket(context, endpoint);
            if (socket == nullptr) {
                result.errnum = zmq_errno();
                return result;
            }
        }

        const int sent = zmq_send(socket, payload.data(), payload.size(), 0);
        if (sent == -1) {
            result.errnum = zmq_errno();
            if (attempt == 0) {
                result.retried = true;
                closeSocket(socket);
                continue;
            }
            result.status = (result.errnum == EAGAIN || result.errnum == EINTR) ? SendStatus::Timeout
                                                                                : SendStatus::Error;
            return result;
        }

        char ack[8] = {};
        const int received = zmq_recv(socket, ack, sizeof(ack), 0);
        if (received == -1) {
            result.errnum = zmq_errno();
            if (attempt == 0) {
                result.retried = true;
                closeSocket(socket);
                continue;
            }
            result.status = (result.errnum == EAGAIN || result.errnum == EINTR) ? SendStatus::Timeout
                                                                                : SendStatus::Error;
            return result;
        }

        if (received < 1) {
            result.status = SendStatus::UnknownAck;
            result.retried = attempt == 0;
            if (attempt == 0) {
                closeSocket(socket);
                continue;
            }
            return result;
        }

        result.ack = ack[0];
        if (ack[0] == '0') {
            result.status = SendStatus::Ok;
            return result;
        }
        if (ack[0] == '1') {
            result.status = SendStatus::ParseFailAck;
            return result;
        }

        if (attempt == 0) {
            result.retried = true;
            closeSocket(socket);
            continue;
        }
        result.status = SendStatus::UnknownAck;
        return result;
    }

    return result;
}

const char* sendStatusText(SendStatus status) {
    switch (status) {
        case SendStatus::Ok:
            return "ACK ok";
        case SendStatus::ParseFailAck:
            return "receiver parse-fail ACK";
        case SendStatus::Timeout:
            return "send/receive timeout";
        case SendStatus::UnknownAck:
            return "unknown ACK";
        case SendStatus::Error:
        default:
            return "socket error";
    }
}

jsa::protocol::Frame3DV1 buildFrame(int frameNumber,
                                    double timestampMs,
                                    const Vector3& headPosition,
                                    bool objectVisible,
                                    int id,
                                    int label) {
    jsa::protocol::Frame3DV1 frame;
    frame.frame_number = frameNumber;
    frame.timestamp_ms = timestampMs;
    if (objectVisible) {
        jsa::protocol::Object3DV1 object;
        object.id = id;
        object.label = label;
        object.x = headPosition.x;
        object.y = -headPosition.y;
        object.z = -headPosition.z;
        frame.objects.push_back(object);
    }
    return frame;
}

ObjectStats computeObjectStats(const Vector3& headPosition) {
    ObjectStats stats;
    const double x = headPosition.x;
    const double y = headPosition.y;
    const double z = headPosition.z;
    const double horizontalDistance = std::sqrt((x * x) + (z * z));
    stats.distanceMeters = std::sqrt((x * x) + (y * y) + (z * z));
    stats.horizontalFovDeg = std::atan2(x, -z) * 180.0 / ANGLE_PI;
    stats.verticalFovDeg = std::atan2(y, horizontalDistance) * 180.0 / ANGLE_PI;
    if (stats.distanceMeters > 0.000001) {
        const double forwardDot = std::clamp(-z / stats.distanceMeters, -1.0, 1.0);
        stats.offCenterDeg = std::acos(forwardDot) * 180.0 / ANGLE_PI;
    }
    return stats;
}

void drawAxes(float axisLength) {
    DrawLine3D(Vector3{0.0f, 0.0f, 0.0f}, Vector3{axisLength, 0.0f, 0.0f}, RED);
    DrawLine3D(Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, axisLength, 0.0f}, GREEN);
    DrawLine3D(Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, 0.0f, axisLength}, BLUE);
    DrawLine3D(Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, 0.0f, -axisLength}, SKYBLUE);
}

void drawForwardMarker() {
    DrawLine3D(Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, 0.0f, -1.0f}, RAYWHITE);
    DrawCylinderEx(Vector3{0.0f, 0.0f, -1.0f},
                   Vector3{0.0f, 0.0f, -1.28f},
                   0.16f,
                   0.0f,
                   16,
                   RAYWHITE);
}

void drawWorldLabel(const char* text, const Vector3& position, const Camera3D& camera) {
    const Vector2 screen = GetWorldToScreen(position, camera);
    DrawText(text, static_cast<int>(screen.x) + 6, static_cast<int>(screen.y) - 6, 14, LIGHTGRAY);
}

void drawRadarPanel(const Vector3& headPosition, bool visible, int width, int height) {
    const int size = 190;
    const int left = std::max(10, width - size - 18);
    const int top = std::max(250, height - size - 18);
    const Vector2 center{static_cast<float>(left + size / 2), static_cast<float>(top + size / 2)};
    constexpr float metersToPixels = 28.0f;

    DrawRectangle(left, top, size, size, Fade(BLACK, 0.55f));
    DrawRectangleLines(left, top, size, size, Fade(RAYWHITE, 0.35f));
    DrawCircleLines(static_cast<int>(center.x), static_cast<int>(center.y), metersToPixels, DARKGRAY);
    DrawCircleLines(static_cast<int>(center.x), static_cast<int>(center.y), metersToPixels * 2.0f, DARKGRAY);
    DrawCircleLines(static_cast<int>(center.x), static_cast<int>(center.y), metersToPixels * 3.0f, DARKGRAY);
    DrawLine(static_cast<int>(center.x), top + 16, static_cast<int>(center.x), top + size - 16, Fade(RAYWHITE, 0.35f));
    DrawLine(left + 16, static_cast<int>(center.y), left + size - 16, static_cast<int>(center.y), Fade(RAYWHITE, 0.35f));
    DrawTriangle(Vector2{center.x, center.y - 12.0f},
                 Vector2{center.x - 7.0f, center.y + 8.0f},
                 Vector2{center.x + 7.0f, center.y + 8.0f},
                 RAYWHITE);
    DrawText("Front", left + size / 2 - 18, top + 4, 12, LIGHTGRAY);
    DrawText("Behind", left + size / 2 - 22, top + size - 18, 12, LIGHTGRAY);
    DrawText("Left", left + 8, top + size / 2 - 6, 12, LIGHTGRAY);
    DrawText("Right", left + size - 42, top + size / 2 - 6, 12, LIGHTGRAY);

    const float px = std::clamp(center.x + headPosition.x * metersToPixels,
                                static_cast<float>(left + 8),
                                static_cast<float>(left + size - 8));
    const float py = std::clamp(center.y + headPosition.z * metersToPixels,
                                static_cast<float>(top + 8),
                                static_cast<float>(top + size - 8));
    DrawCircleV(Vector2{px, py}, 6.0f, visible ? ORANGE : Fade(ORANGE, 0.28f));
}

bool pointInRect(Vector2 point, Rectangle rect) {
    return point.x >= rect.x && point.x <= rect.x + rect.width && point.y >= rect.y &&
           point.y <= rect.y + rect.height;
}

} // namespace

int main(int argc, char* argv[]) {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    const CLIOptions options = parseCommandLine(argc, argv);
    if (options.showHelp) {
        printUsage(argv[0]);
        return 0;
    }
    if (!options.valid) {
        printUsage(argv[0]);
        return 1;
    }

    void* context = zmq_ctx_new();
    if (context == nullptr) {
        std::cerr << "Error: failed to create ZeroMQ context\n";
        return 1;
    }
    void* socket = createReqSocket(context, options.ipcEndpoint);
    if (socket == nullptr) {
        zmq_ctx_term(context);
        return 1;
    }

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(options.width, options.height, "jsa-object-sim");
    SetTargetFPS(static_cast<int>(std::round(options.fps)));

    Camera3D camera{};
    const Camera3D defaultCamera{Vector3{6.5f, 5.0f, 6.5f},
                                 Vector3{0.0f, 0.0f, -1.0f},
                                 Vector3{0.0f, 1.0f, 0.0f},
                                 60.0f,
                                 CAMERA_PERSPECTIVE};
    camera = defaultCamera;

    Vector3 objectPosition = options.initialHeadPosition;
    bool objectVisible = true;
    Stats stats;
    SendResult lastSend;
    std::string lastStatus = "not sent yet";
    int frameNumber = 0;
    const auto start = std::chrono::steady_clock::now();
    auto nextTick = start;
    const auto frameInterval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / options.fps));

    while (gRunning.load() && !WindowShouldClose()) {
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            UpdateCamera(&camera, CAMERA_FREE);
        }

        const float step = (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT))
                               ? FAST_STEP_METERS
                               : ((IsKeyDown(KEY_LEFT_ALT) || IsKeyDown(KEY_RIGHT_ALT)) ? SLOW_STEP_METERS
                                                                                        : BASE_STEP_METERS);
        if (IsKeyPressed(KEY_LEFT)) {
            objectPosition.x -= step;
        }
        if (IsKeyPressed(KEY_RIGHT)) {
            objectPosition.x += step;
        }
        if (IsKeyPressed(KEY_UP)) {
            objectPosition.z -= step;
        }
        if (IsKeyPressed(KEY_DOWN)) {
            objectPosition.z += step;
        }
        if (IsKeyPressed(KEY_PAGE_UP)) {
            objectPosition.y += step;
        }
        if (IsKeyPressed(KEY_PAGE_DOWN)) {
            objectPosition.y -= step;
        }

        if (IsKeyPressed(KEY_ONE)) {
            objectPosition = Vector3{0.0f, 0.0f, -1.0f};
        } else if (IsKeyPressed(KEY_TWO)) {
            objectPosition = Vector3{0.0f, 0.0f, 1.0f};
        } else if (IsKeyPressed(KEY_THREE)) {
            objectPosition = Vector3{-1.0f, 0.0f, 0.0f};
        } else if (IsKeyPressed(KEY_FOUR)) {
            objectPosition = Vector3{1.0f, 0.0f, 0.0f};
        } else if (IsKeyPressed(KEY_FIVE)) {
            objectPosition = Vector3{0.0f, 1.0f, 0.0f};
        } else if (IsKeyPressed(KEY_SIX)) {
            objectPosition = Vector3{0.0f, -1.0f, 0.0f};
        }

        if (IsKeyPressed(KEY_SPACE)) {
            objectVisible = !objectVisible;
        }
        if (IsKeyPressed(KEY_R)) {
            objectPosition = options.initialHeadPosition;
            objectVisible = true;
            camera = defaultCamera;
        }

        const Rectangle toggleButton{16.0f, 16.0f, 142.0f, 34.0f};
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && pointInRect(GetMousePosition(), toggleButton)) {
            objectVisible = !objectVisible;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= nextTick) {
            const double timestampMs =
                std::chrono::duration<double, std::milli>(now - start).count();
            jsa::protocol::Frame3DV1 frame = buildFrame(frameNumber,
                                                        timestampMs,
                                                        objectPosition,
                                                        objectVisible,
                                                        options.id,
                                                        options.label);

            std::vector<uint8_t> payload;
            std::string err;
            if (!jsa::protocol::serializeFrame3DV1(frame, payload, err)) {
                ++stats.otherErrors;
                lastStatus = "serialize error: " + err;
            } else {
                ++stats.sent;
                lastSend = sendFrameWithSingleRetry(socket, context, options.ipcEndpoint, payload);
                if (lastSend.retried) {
                    ++stats.retries;
                }
                if (lastSend.status == SendStatus::Ok) {
                    ++stats.ackedOk;
                } else if (lastSend.status == SendStatus::ParseFailAck) {
                    ++stats.ackedFail;
                } else if (lastSend.status == SendStatus::Timeout) {
                    ++stats.timeouts;
                } else {
                    ++stats.otherErrors;
                }
                lastStatus = sendStatusText(lastSend.status);
                if (lastSend.errnum != 0) {
                    lastStatus += ": ";
                    lastStatus += zmq_strerror(lastSend.errnum);
                } else if (lastSend.status == SendStatus::UnknownAck) {
                    lastStatus += " ";
                    lastStatus += lastSend.ack;
                }
            }

            ++frameNumber;
            nextTick += frameInterval;
            if (now - nextTick > frameInterval * 5) {
                nextTick = now + frameInterval;
            }
        }

        BeginDrawing();
        ClearBackground(Color{16, 19, 24, 255});

        BeginMode3D(camera);
        DrawGrid(40, 1.0f);
        drawAxes(2.5f);
        drawForwardMarker();
        DrawSphere(Vector3{0.0f, 0.0f, 0.0f}, 0.06f, WHITE);
        DrawLine3D(Vector3{0.0f, 0.0f, 0.0f}, objectPosition, Fade(ORANGE, objectVisible ? 0.6f : 0.2f));
        DrawSphere(objectPosition, 0.15f, objectVisible ? ORANGE : Fade(ORANGE, 0.25f));
        DrawSphereWires(objectPosition, 0.16f, 8, 8, Fade(WHITE, objectVisible ? 0.35f : 0.15f));
        EndMode3D();

        drawWorldLabel("Right +X", Vector3{2.55f, 0.0f, 0.0f}, camera);
        drawWorldLabel("Up +Y", Vector3{0.0f, 2.55f, 0.0f}, camera);
        drawWorldLabel("Behind +Z", Vector3{0.0f, 0.0f, 2.55f}, camera);
        drawWorldLabel("Forward -Z", Vector3{0.0f, 0.0f, -2.55f}, camera);
        drawWorldLabel("Object", Vector3{objectPosition.x, objectPosition.y + 0.28f, objectPosition.z}, camera);

        drawRadarPanel(objectPosition, objectVisible, options.width, options.height);

        DrawRectangleRec(toggleButton, objectVisible ? Fade(DARKGREEN, 0.92f) : Fade(MAROON, 0.92f));
        DrawRectangleLinesEx(toggleButton, 1.0f, Fade(RAYWHITE, 0.45f));
        DrawText(objectVisible ? "Object: On" : "Object: Off",
                 static_cast<int>(toggleButton.x) + 13,
                 static_cast<int>(toggleButton.y) + 9,
                 16,
                 RAYWHITE);

        const ObjectStats objectStats = computeObjectStats(objectPosition);
        DrawRectangle(16, 66, 760, 220, Fade(BLACK, 0.72f));
        DrawRectangleLines(16, 66, 760, 220, Fade(RAYWHITE, 0.42f));
        DrawText("jsa-object-sim", 30, 80, 18, RAYWHITE);
        char line[256];
        std::snprintf(line,
                      sizeof(line),
                      "Endpoint: %s | FPS: %.1f | id: %d label: %d",
                      options.ipcEndpoint.c_str(),
                      options.fps,
                      options.id,
                      options.label);
        DrawText(line, 30, 108, 14, LIGHTGRAY);

        DrawRectangle(28, 126, 720, 78, Fade(Color{38, 48, 58, 255}, 0.88f));
        DrawRectangleLines(28, 126, 720, 78, Fade(RAYWHITE, 0.28f));
        std::snprintf(line,
                      sizeof(line),
                      "Head XYZ: x=%+.2f y=%+.2f z=%+.2f m",
                      objectPosition.x,
                      objectPosition.y,
                      objectPosition.z);
        DrawText(line, 42, 136, 16, RAYWHITE);
        std::snprintf(line,
                      sizeof(line),
                      "Socket XYZ: x=%+.2f y=%+.2f z=%+.2f m",
                      objectPosition.x,
                      -objectPosition.y,
                      -objectPosition.z);
        DrawText(line, 42, 160, 16, RAYWHITE);
        std::snprintf(line,
                      sizeof(line),
                      "Distance: %.2f m | FOV from center: %.1f deg | horizontal: %+.1f deg | vertical: %+.1f deg",
                      objectStats.distanceMeters,
                      objectStats.offCenterDeg,
                      objectStats.horizontalFovDeg,
                      objectStats.verticalFovDeg);
        DrawText(line, 42, 184, 16, GOLD);
        std::snprintf(line,
                      sizeof(line),
                      "Frame: %d | object_count: %d | status: %s",
                      frameNumber,
                      objectVisible ? 1 : 0,
                      lastStatus.c_str());
        DrawText(line, 30, 218, 14, lastSend.status == SendStatus::Ok ? LIME : YELLOW);
        std::snprintf(line,
                      sizeof(line),
                      "Stats ok=%llu fail=%llu timeout=%llu err=%llu retry=%llu",
                      static_cast<unsigned long long>(stats.ackedOk),
                      static_cast<unsigned long long>(stats.ackedFail),
                      static_cast<unsigned long long>(stats.timeouts),
                      static_cast<unsigned long long>(stats.otherErrors),
                      static_cast<unsigned long long>(stats.retries));
        DrawText(line, 30, 240, 14, LIGHTGRAY);
        DrawText("Arrows move X/Z | PgUp/PgDn Y | Shift/Alt step | 1-6 presets | Space toggle | R reset | right/middle mouse camera",
                 30,
                 262,
                 14,
                 LIGHTGRAY);

        EndDrawing();
    }

    closeSocket(socket);
    zmq_ctx_term(context);
    CloseWindow();
    return 0;
}
