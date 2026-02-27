#include <raylib.h>
#include <zmq.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

namespace {

constexpr const char* DEFAULT_IPC_ENDPOINT = "ipc:///tmp/jv/audio/0.sock";
constexpr const char* DEFAULT_FORWARD_IPC_ENDPOINT = "ipc:///tmp/jv/audio/1.sock";
constexpr int DEFAULT_WINDOW_WIDTH = 1280;
constexpr int DEFAULT_WINDOW_HEIGHT = 720;
constexpr int DEFAULT_TARGET_FPS = 60;
constexpr float DEFAULT_TRAIL_SECONDS = 2.0f;

constexpr int SOCKET_LINGER_MS = 0;
constexpr int UPSTREAM_RECV_TIMEOUT_MS = 2;
constexpr int DEFAULT_FORWARD_SEND_TIMEOUT_MS = 100;
constexpr int DEFAULT_FORWARD_RECV_TIMEOUT_MS = 100;
constexpr int DEFAULT_FORWARD_RETRIES = 1;
constexpr int MAX_MESSAGES_PER_TICK = 4;

constexpr double STALE_FADE_START_SECONDS = 0.5;
constexpr double STALE_DROP_SECONDS = 2.0;
constexpr size_t MAX_TRAIL_POINTS_PER_OBJECT = 2048;
constexpr int32_t MAX_OBJECTS_SAFETY_LIMIT = 100000;

std::atomic<bool> gRunning{true};

void handleSignal(int) {
    gRunning.store(false);
}

struct CLIOptions {
    std::string ipcEndpoint = DEFAULT_IPC_ENDPOINT;
    std::string forwardIpcEndpoint = DEFAULT_FORWARD_IPC_ENDPOINT;
    bool noForward = false;
    int forwardSendTimeoutMs = DEFAULT_FORWARD_SEND_TIMEOUT_MS;
    int forwardRecvTimeoutMs = DEFAULT_FORWARD_RECV_TIMEOUT_MS;
    int forwardRetries = DEFAULT_FORWARD_RETRIES;
    int width = DEFAULT_WINDOW_WIDTH;
    int height = DEFAULT_WINDOW_HEIGHT;
    int fps = DEFAULT_TARGET_FPS;
    float trailSeconds = DEFAULT_TRAIL_SECONDS;
    bool showHelp = false;
    bool valid = true;
};

struct OrbitObject {
    int id = -1;
    int label = -1;
    double x = 0.0;
    double y = 0.0;
    double z = -1.0;
};

struct OrbitFrame {
    int frameNumber = 0;
    double timestampMs = 0.0;
    std::vector<OrbitObject> objects;
};

struct TrailPoint {
    Vector3 position{};
    double timeSec = 0.0;
};

struct ObjectState {
    int id = -1;
    int label = -1;
    Vector3 position{};
    double lastSeenSec = 0.0;
    std::deque<TrailPoint> trail;
};

struct RuntimeStats {
    uint64_t receivedFrames = 0;
    uint64_t parseFailures = 0;
    uint64_t forwardAttempts = 0;
    uint64_t forwardOk = 0;
    uint64_t forwardSendFail = 0;
    uint64_t forwardAckFail = 0;
    uint64_t forwardSendTimeout = 0;
    uint64_t forwardSendError = 0;
    uint64_t forwardAckTimeout = 0;
    uint64_t forwardAckError = 0;
    uint64_t forwardAckUnexpected = 0;
    uint64_t forwardReconnects = 0;
};

class FpsEstimator {
public:
    void noteFrame(double nowSec) {
        samplesSec.push_back(nowSec);
        while (!samplesSec.empty() && (nowSec - samplesSec.front()) > 1.0) {
            samplesSec.pop_front();
        }
    }

    double fps() const {
        if (samplesSec.empty()) {
            return 0.0;
        }
        if (samplesSec.size() < 2) {
            return static_cast<double>(samplesSec.size());
        }
        const double dt = samplesSec.back() - samplesSec.front();
        if (dt <= 1e-6) {
            return 0.0;
        }
        return static_cast<double>(samplesSec.size() - 1) / dt;
    }

private:
    std::deque<double> samplesSec;
};

enum class ForwardResult {
    Ok,
    SendTimeout,
    SendError,
    AckTimeout,
    AckError,
    AckUnexpected,
};

double nowSteadySec() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

void printUsage(const char* executableName) {
    std::cout << "Usage: " << executableName << " [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --ipc <endpoint>          Upstream ZeroMQ endpoint (default: "
              << DEFAULT_IPC_ENDPOINT << ")\n";
    std::cout << "  --forward-ipc <endpoint>  Downstream forward endpoint (default: "
              << DEFAULT_FORWARD_IPC_ENDPOINT << ")\n";
    std::cout << "  --no-forward              Disable forwarding to downstream consumer\n";
    std::cout << "  --forward-send-timeout-ms <ms>  Forward send timeout (default: "
              << DEFAULT_FORWARD_SEND_TIMEOUT_MS << ")\n";
    std::cout << "  --forward-recv-timeout-ms <ms>  Forward ACK timeout (default: "
              << DEFAULT_FORWARD_RECV_TIMEOUT_MS << ")\n";
    std::cout << "  --forward-retries <count>       Forward retry count (default: "
              << DEFAULT_FORWARD_RETRIES << ")\n";
    std::cout << "  --width <pixels>          Window width (default: " << DEFAULT_WINDOW_WIDTH << ")\n";
    std::cout << "  --height <pixels>         Window height (default: " << DEFAULT_WINDOW_HEIGHT
              << ")\n";
    std::cout << "  --fps <value>             Render target FPS (default: " << DEFAULT_TARGET_FPS
              << ")\n";
    std::cout << "  --trail-seconds <value>   Trail duration in seconds (default: "
              << DEFAULT_TRAIL_SECONDS << ")\n";
    std::cout << "  --help, -h                Show this help message\n";
}

bool parseIntArg(const char* text, int& outValue) {
    if (text == nullptr) {
        return false;
    }

    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    if (value < static_cast<long>(std::numeric_limits<int32_t>::min()) ||
        value > static_cast<long>(std::numeric_limits<int32_t>::max())) {
        return false;
    }

    outValue = static_cast<int>(value);
    return true;
}

bool parseFloatArg(const char* text, float& outValue) {
    if (text == nullptr) {
        return false;
    }

    char* end = nullptr;
    const float value = std::strtof(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) {
        return false;
    }

    outValue = value;
    return true;
}

CLIOptions parseCommandLine(int argc, char* argv[]) {
    CLIOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            options.showHelp = true;
            continue;
        }

        auto requireValue = [&](const char* optionName) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for option: " << optionName << "\n";
                options.valid = false;
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--ipc") {
            const char* value = requireValue("--ipc");
            if (value != nullptr) {
                options.ipcEndpoint = value;
            }
            continue;
        }
        if (arg == "--forward-ipc") {
            const char* value = requireValue("--forward-ipc");
            if (value != nullptr) {
                options.forwardIpcEndpoint = value;
            }
            continue;
        }
        if (arg == "--no-forward") {
            options.noForward = true;
            continue;
        }
        if (arg == "--forward-send-timeout-ms") {
            const char* value = requireValue("--forward-send-timeout-ms");
            if (value != nullptr && !parseIntArg(value, options.forwardSendTimeoutMs)) {
                std::cerr << "Invalid --forward-send-timeout-ms value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--forward-recv-timeout-ms") {
            const char* value = requireValue("--forward-recv-timeout-ms");
            if (value != nullptr && !parseIntArg(value, options.forwardRecvTimeoutMs)) {
                std::cerr << "Invalid --forward-recv-timeout-ms value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--forward-retries") {
            const char* value = requireValue("--forward-retries");
            if (value != nullptr && !parseIntArg(value, options.forwardRetries)) {
                std::cerr << "Invalid --forward-retries value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--width") {
            const char* value = requireValue("--width");
            if (value != nullptr && !parseIntArg(value, options.width)) {
                std::cerr << "Invalid --width value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--height") {
            const char* value = requireValue("--height");
            if (value != nullptr && !parseIntArg(value, options.height)) {
                std::cerr << "Invalid --height value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--fps") {
            const char* value = requireValue("--fps");
            if (value != nullptr && !parseIntArg(value, options.fps)) {
                std::cerr << "Invalid --fps value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--trail-seconds") {
            const char* value = requireValue("--trail-seconds");
            if (value != nullptr && !parseFloatArg(value, options.trailSeconds)) {
                std::cerr << "Invalid --trail-seconds value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }

        std::cerr << "Unknown option: " << arg << "\n";
        options.valid = false;
    }

    if (options.width <= 0) {
        std::cerr << "--width must be > 0\n";
        options.valid = false;
    }
    if (options.height <= 0) {
        std::cerr << "--height must be > 0\n";
        options.valid = false;
    }
    if (options.fps <= 0) {
        std::cerr << "--fps must be > 0\n";
        options.valid = false;
    }
    if (options.forwardSendTimeoutMs <= 0) {
        std::cerr << "--forward-send-timeout-ms must be > 0\n";
        options.valid = false;
    }
    if (options.forwardRecvTimeoutMs <= 0) {
        std::cerr << "--forward-recv-timeout-ms must be > 0\n";
        options.valid = false;
    }
    if (options.forwardRetries < 0) {
        std::cerr << "--forward-retries must be >= 0\n";
        options.valid = false;
    }
    if (!(options.trailSeconds > 0.0f)) {
        std::cerr << "--trail-seconds must be > 0\n";
        options.valid = false;
    }

    return options;
}

void prepareIpcEndpoint(const std::string& endpoint) {
    const std::string prefix = "ipc://";
    if (endpoint.rfind(prefix, 0) != 0) {
        return;
    }

    const std::string socketPath = endpoint.substr(prefix.size());
    const size_t slashPos = socketPath.find_last_of('/');
    if (slashPos != std::string::npos) {
        const std::string directory = socketPath.substr(0, slashPos);
        if (!directory.empty()) {
            size_t cursor = 1;
            while (cursor != std::string::npos) {
                cursor = directory.find('/', cursor);
                const std::string partial = directory.substr(0, cursor);
                if (!partial.empty()) {
                    ::mkdir(partial.c_str(), 0777);
                }
                if (cursor != std::string::npos) {
                    ++cursor;
                }
            }
        }
    }

    std::remove(socketPath.c_str());
}

void closeSocket(void*& socket) {
    if (socket != nullptr) {
        zmq_close(socket);
        socket = nullptr;
    }
}

void* createUpstreamRepSocket(void* context, const std::string& endpoint) {
    void* socket = zmq_socket(context, ZMQ_REP);
    if (socket == nullptr) {
        std::cerr << "Error: failed to create ZMQ_REP socket: "
                  << zmq_strerror(zmq_errno()) << "\n";
        return nullptr;
    }

    const int linger = SOCKET_LINGER_MS;
    const int timeout = UPSTREAM_RECV_TIMEOUT_MS;
    zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));

    if (zmq_bind(socket, endpoint.c_str()) != 0) {
        std::cerr << "Error: failed to bind upstream socket " << endpoint << ": "
                  << zmq_strerror(zmq_errno()) << "\n";
        zmq_close(socket);
        return nullptr;
    }

    return socket;
}

bool isTimeoutErrno(int errnum) {
    return errnum == EAGAIN || errnum == EINTR;
}

void* createForwardReqSocket(void* context,
                             const std::string& endpoint,
                             int sendTimeoutMs,
                             int recvTimeoutMs) {
    void* socket = zmq_socket(context, ZMQ_REQ);
    if (socket == nullptr) {
        std::cerr << "Warning: failed to create forward ZMQ_REQ socket: "
                  << zmq_strerror(zmq_errno()) << "\n";
        return nullptr;
    }

    const int linger = SOCKET_LINGER_MS;
    const int sendTimeout = sendTimeoutMs;
    const int recvTimeout = recvTimeoutMs;
    zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(socket, ZMQ_SNDTIMEO, &sendTimeout, sizeof(sendTimeout));
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &recvTimeout, sizeof(recvTimeout));

    if (zmq_connect(socket, endpoint.c_str()) != 0) {
        std::cerr << "Warning: failed to connect forward socket " << endpoint << ": "
                  << zmq_strerror(zmq_errno()) << "\n";
        zmq_close(socket);
        return nullptr;
    }

    return socket;
}

bool readInt32(const uint8_t* data, size_t len, size_t& offset, int32_t& outValue) {
    if (offset + sizeof(int32_t) > len) {
        return false;
    }
    std::memcpy(&outValue, data + offset, sizeof(int32_t));
    offset += sizeof(int32_t);
    return true;
}

bool readDouble(const uint8_t* data, size_t len, size_t& offset, double& outValue) {
    if (offset + sizeof(double) > len) {
        return false;
    }
    std::memcpy(&outValue, data + offset, sizeof(double));
    offset += sizeof(double);
    return true;
}

bool readExpectedChar(const uint8_t* data, size_t len, size_t& offset, char expected) {
    if (offset >= len) {
        return false;
    }
    if (static_cast<char>(data[offset]) != expected) {
        return false;
    }
    ++offset;
    return true;
}

bool parseOrbitPayload(const uint8_t* data,
                       size_t len,
                       OrbitFrame& outFrame,
                       std::string& errorMessage) {
    outFrame = {};
    errorMessage.clear();

    if (data == nullptr) {
        errorMessage = "Input payload pointer is null.";
        return false;
    }
    if (len < sizeof(int32_t) + sizeof(double) + 1 + sizeof(int32_t)) {
        errorMessage = "Payload too short.";
        return false;
    }

    size_t offset = 0;
    int32_t frameNumber = 0;
    double timestampMs = 0.0;
    if (!readInt32(data, len, offset, frameNumber)) {
        errorMessage = "Failed to read frame_number.";
        return false;
    }
    if (!readDouble(data, len, offset, timestampMs)) {
        errorMessage = "Failed to read timestamp_ms.";
        return false;
    }
    if (!readExpectedChar(data, len, offset, '^')) {
        errorMessage = "Missing list start marker '^'.";
        return false;
    }

    int32_t objectCount = 0;
    if (!readInt32(data, len, offset, objectCount)) {
        errorMessage = "Failed to read object count.";
        return false;
    }
    if (objectCount < 0) {
        errorMessage = "Object count is negative.";
        return false;
    }
    if (objectCount > MAX_OBJECTS_SAFETY_LIMIT) {
        errorMessage = "Object count exceeds safety limit.";
        return false;
    }

    outFrame.frameNumber = frameNumber;
    outFrame.timestampMs = timestampMs;
    outFrame.objects.clear();
    outFrame.objects.reserve(static_cast<size_t>(objectCount));

    for (int32_t i = 0; i < objectCount; ++i) {
        if (!readExpectedChar(data, len, offset, '|')) {
            errorMessage = "Missing object delimiter '|'.";
            return false;
        }

        OrbitObject object;
        int32_t id = -1;
        int32_t label = -1;
        double x = 0.0;
        double y = 0.0;
        double z = -1.0;

        if (!readInt32(data, len, offset, id)) {
            errorMessage = "Failed to read object id.";
            return false;
        }
        if (!readInt32(data, len, offset, label)) {
            errorMessage = "Failed to read object label.";
            return false;
        }
        if (!readDouble(data, len, offset, x)) {
            errorMessage = "Failed to read object x.";
            return false;
        }
        if (!readDouble(data, len, offset, y)) {
            errorMessage = "Failed to read object y.";
            return false;
        }
        if (!readDouble(data, len, offset, z)) {
            errorMessage = "Failed to read object z.";
            return false;
        }

        object.id = id;
        object.label = label;
        object.x = x;
        object.y = y;
        object.z = z;
        outFrame.objects.push_back(object);
    }

    if (offset < len && static_cast<char>(data[offset]) == '^') {
        ++offset;
    }
    if (offset != len) {
        errorMessage = "Trailing bytes found after payload parse.";
        return false;
    }

    return true;
}

ForwardResult forwardPayloadWithRetry(void*& forwardSocket,
                                      void* context,
                                      const std::string& endpoint,
                                      int sendTimeoutMs,
                                      int recvTimeoutMs,
                                      int retries,
                                      const uint8_t* payload,
                                      size_t payloadLen,
                                      uint64_t& reconnectCounter) {
    ForwardResult lastResult = ForwardResult::SendError;
    const int totalAttempts = std::max(1, retries + 1);

    for (int attempt = 0; attempt < totalAttempts; ++attempt) {
        if (forwardSocket == nullptr) {
            forwardSocket = createForwardReqSocket(
                context, endpoint, sendTimeoutMs, recvTimeoutMs);
            if (forwardSocket == nullptr) {
                lastResult = ForwardResult::SendError;
                return lastResult;
            }
        }

        const int sent = zmq_send(forwardSocket, payload, payloadLen, 0);
        if (sent == -1) {
            lastResult = isTimeoutErrno(zmq_errno()) ? ForwardResult::SendTimeout
                                                     : ForwardResult::SendError;
            closeSocket(forwardSocket);
            if (attempt + 1 < totalAttempts) {
                ++reconnectCounter;
                continue;
            }
            return lastResult;
        }

        char ack[8] = {};
        const int recvBytes = zmq_recv(forwardSocket, ack, sizeof(ack), 0);
        if (recvBytes == -1) {
            lastResult = isTimeoutErrno(zmq_errno()) ? ForwardResult::AckTimeout
                                                     : ForwardResult::AckError;
            closeSocket(forwardSocket);
            if (attempt + 1 < totalAttempts) {
                ++reconnectCounter;
                continue;
            }
            return lastResult;
        }
        if (recvBytes < 1 || ack[0] != '0') {
            lastResult = ForwardResult::AckUnexpected;
            closeSocket(forwardSocket);
            if (attempt + 1 < totalAttempts) {
                ++reconnectCounter;
                continue;
            }
            return lastResult;
        }

        return ForwardResult::Ok;
    }

    return lastResult;
}

float computeObjectFade(double ageSec) {
    if (ageSec <= STALE_FADE_START_SECONDS) {
        return 1.0f;
    }
    if (ageSec >= STALE_DROP_SECONDS) {
        return 0.0f;
    }
    const double fadeWindow = STALE_DROP_SECONDS - STALE_FADE_START_SECONDS;
    return static_cast<float>((STALE_DROP_SECONDS - ageSec) / fadeWindow);
}

Color colorForObjectId(int objectId) {
    const uint32_t h = static_cast<uint32_t>(std::hash<int>{}(objectId));
    const uint8_t r = static_cast<uint8_t>(80u + (h & 0x7Fu));
    const uint8_t g = static_cast<uint8_t>(80u + ((h >> 8) & 0x7Fu));
    const uint8_t b = static_cast<uint8_t>(80u + ((h >> 16) & 0x7Fu));
    return Color{r, g, b, 255};
}

void pruneTrail(ObjectState& state, double nowSec, double trailSeconds) {
    while (!state.trail.empty() && (nowSec - state.trail.front().timeSec) > trailSeconds) {
        state.trail.pop_front();
    }
    while (state.trail.size() > MAX_TRAIL_POINTS_PER_OBJECT) {
        state.trail.pop_front();
    }
}

void upsertObjectsFromFrame(const OrbitFrame& frame,
                            double nowSec,
                            std::unordered_map<int, ObjectState>& objects,
                            double trailSeconds) {
    for (const OrbitObject& object : frame.objects) {
        if (!std::isfinite(object.x) || !std::isfinite(object.y) || !std::isfinite(object.z)) {
            continue;
        }

        auto it = objects.find(object.id);
        if (it == objects.end()) {
            ObjectState state;
            state.id = object.id;
            state.label = object.label;
            state.position = Vector3{
                static_cast<float>(object.x),
                static_cast<float>(object.y),
                static_cast<float>(object.z)};
            state.lastSeenSec = nowSec;
            state.trail.push_back(TrailPoint{state.position, nowSec});
            objects.emplace(object.id, std::move(state));
            continue;
        }

        ObjectState& state = it->second;
        state.label = object.label;
        state.position = Vector3{
            static_cast<float>(object.x),
            static_cast<float>(object.y),
            static_cast<float>(object.z)};
        state.lastSeenSec = nowSec;
        state.trail.push_back(TrailPoint{state.position, nowSec});
        pruneTrail(state, nowSec, trailSeconds);
    }
}

void pruneStaleObjects(std::unordered_map<int, ObjectState>& objects,
                       double nowSec,
                       double trailSeconds) {
    for (auto it = objects.begin(); it != objects.end();) {
        ObjectState& state = it->second;
        pruneTrail(state, nowSec, trailSeconds);
        const double ageSec = nowSec - state.lastSeenSec;
        if (ageSec > STALE_DROP_SECONDS) {
            it = objects.erase(it);
        } else {
            ++it;
        }
    }
}

void drawAxes(float axisLength) {
    DrawLine3D(Vector3{0.0f, 0.0f, 0.0f}, Vector3{axisLength, 0.0f, 0.0f}, RED);
    DrawLine3D(Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, axisLength, 0.0f}, GREEN);
    DrawLine3D(Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, 0.0f, axisLength}, BLUE);
}

float clamp01(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
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

    prepareIpcEndpoint(options.ipcEndpoint);

    void* zmqContext = zmq_ctx_new();
    if (zmqContext == nullptr) {
        std::cerr << "Error: failed to create ZeroMQ context\n";
        return 1;
    }

    void* upstreamSocket = createUpstreamRepSocket(zmqContext, options.ipcEndpoint);
    if (upstreamSocket == nullptr) {
        zmq_ctx_term(zmqContext);
        return 1;
    }

    void* forwardSocket = nullptr;
    if (!options.noForward) {
        forwardSocket = createForwardReqSocket(zmqContext,
                                               options.forwardIpcEndpoint,
                                               options.forwardSendTimeoutMs,
                                               options.forwardRecvTimeoutMs);
        if (forwardSocket == nullptr) {
            std::cerr << "Warning: continuing without an active downstream connection; "
                      << "will retry during runtime.\n";
        }
    }

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(options.width, options.height, "render-visual-stream");
    SetTargetFPS(options.fps);

    Camera3D camera{};
    camera.position = Vector3{6.5f, 5.0f, 6.5f};
    camera.target = Vector3{0.0f, 0.0f, -2.0f};
    camera.up = Vector3{0.0f, 1.0f, 0.0f};
    camera.fovy = 60.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    std::unordered_map<int, ObjectState> objects;
    RuntimeStats stats;
    FpsEstimator fpsEstimator;

    int latestFrameNumber = 0;
    double latestTimestampMs = 0.0;
    size_t latestObjectCount = 0;
    bool hasFrame = false;
    uint64_t renderFrameCounter = 0;

    std::cout << "Listening on " << options.ipcEndpoint << "\n";
    if (options.noForward) {
        std::cout << "Forwarding disabled\n";
    } else {
        std::cout << "Forwarding to " << options.forwardIpcEndpoint << "\n";
        std::cout << "Forward policy: send-timeout=" << options.forwardSendTimeoutMs
                  << " ms, recv-timeout=" << options.forwardRecvTimeoutMs
                  << " ms, retries=" << options.forwardRetries << "\n";
    }
    std::cout << "Press Ctrl+C or close the window to stop.\n";

    while (gRunning.load() && !WindowShouldClose()) {
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            UpdateCamera(&camera, CAMERA_FREE);
        }
        if (IsKeyPressed(KEY_R)) {
            camera.position = Vector3{6.5f, 5.0f, 6.5f};
            camera.target = Vector3{0.0f, 0.0f, -2.0f};
            camera.up = Vector3{0.0f, 1.0f, 0.0f};
        }

        for (int i = 0; i < MAX_MESSAGES_PER_TICK && gRunning.load(); ++i) {
            zmq_msg_t message;
            zmq_msg_init(&message);

            const int recvResult = zmq_msg_recv(&message, upstreamSocket, 0);
            if (recvResult == -1) {
                const int errnum = zmq_errno();
                zmq_msg_close(&message);

                if (errnum == EAGAIN || errnum == EINTR) {
                    break;
                }

                std::cerr << "Error: upstream receive failed: " << zmq_strerror(errnum) << "\n";
                gRunning.store(false);
                break;
            }

            const auto* payload = static_cast<const uint8_t*>(zmq_msg_data(&message));
            const size_t payloadLen = zmq_msg_size(&message);

            OrbitFrame frame;
            std::string parseError;
            const bool parsed = parseOrbitPayload(payload, payloadLen, frame, parseError);

            const char ack = parsed ? '0' : '1';
            if (zmq_send(upstreamSocket, &ack, 1, 0) == -1) {
                std::cerr << "Error: failed to send upstream ACK: "
                          << zmq_strerror(zmq_errno()) << "\n";
                zmq_msg_close(&message);
                gRunning.store(false);
                break;
            }

            if (!parsed) {
                ++stats.parseFailures;
                if (stats.parseFailures % 30 == 1) {
                    std::cerr << "Parse failure (" << stats.parseFailures << "): "
                              << parseError << "\n";
                }
                zmq_msg_close(&message);
                continue;
            }

            const double nowSec = nowSteadySec();
            upsertObjectsFromFrame(frame, nowSec, objects, options.trailSeconds);

            ++stats.receivedFrames;
            fpsEstimator.noteFrame(nowSec);
            latestFrameNumber = frame.frameNumber;
            latestTimestampMs = frame.timestampMs;
            latestObjectCount = frame.objects.size();
            hasFrame = true;

            if (!options.noForward) {
                ++stats.forwardAttempts;
                const ForwardResult result = forwardPayloadWithRetry(
                    forwardSocket,
                    zmqContext,
                    options.forwardIpcEndpoint,
                    options.forwardSendTimeoutMs,
                    options.forwardRecvTimeoutMs,
                    options.forwardRetries,
                    payload,
                    payloadLen,
                    stats.forwardReconnects);

                if (result == ForwardResult::Ok) {
                    ++stats.forwardOk;
                } else if (result == ForwardResult::SendTimeout) {
                    ++stats.forwardSendFail;
                    ++stats.forwardSendTimeout;
                } else if (result == ForwardResult::SendError) {
                    ++stats.forwardSendFail;
                    ++stats.forwardSendError;
                } else if (result == ForwardResult::AckTimeout) {
                    ++stats.forwardAckFail;
                    ++stats.forwardAckTimeout;
                } else if (result == ForwardResult::AckError) {
                    ++stats.forwardAckFail;
                    ++stats.forwardAckError;
                } else {
                    ++stats.forwardAckFail;
                    ++stats.forwardAckUnexpected;
                }
            }

            zmq_msg_close(&message);
        }

        const double nowSec = nowSteadySec();
        pruneStaleObjects(objects, nowSec, options.trailSeconds);

        BeginDrawing();
        ClearBackground(Color{16, 19, 24, 255});

        BeginMode3D(camera);
        DrawGrid(40, 1.0f);
        drawAxes(2.5f);
        DrawSphere(Vector3{0.0f, 0.0f, 0.0f}, 0.06f, WHITE);

        for (const auto& entry : objects) {
            const ObjectState& state = entry.second;
            const double ageSec = nowSec - state.lastSeenSec;
            const float fade = computeObjectFade(ageSec);
            if (fade <= 0.0f) {
                continue;
            }

            const Color base = colorForObjectId(state.id);

            if (state.trail.size() > 1) {
                for (size_t idx = 1; idx < state.trail.size(); ++idx) {
                    const TrailPoint& a = state.trail[idx - 1];
                    const TrailPoint& b = state.trail[idx];
                    const float trailFade = clamp01(
                        1.0f - static_cast<float>((nowSec - b.timeSec) / options.trailSeconds));
                    const float alpha = fade * trailFade * 0.9f;
                    if (alpha <= 0.0f) {
                        continue;
                    }
                    Color trailColor = base;
                    trailColor.a = static_cast<unsigned char>(std::round(alpha * 255.0f));
                    DrawLine3D(a.position, b.position, trailColor);
                }
            }

            Color fill = base;
            fill.a = static_cast<unsigned char>(std::round(fade * 255.0f));
            DrawSphere(state.position, 0.14f, fill);
            DrawSphereWires(state.position, 0.15f, 8, 8, Fade(WHITE, 0.2f * fade));
        }
        EndMode3D();

        for (const auto& entry : objects) {
            const ObjectState& state = entry.second;
            const double ageSec = nowSec - state.lastSeenSec;
            const float fade = computeObjectFade(ageSec);
            if (fade <= 0.0f) {
                continue;
            }

            const Vector3 labelAnchor{
                state.position.x,
                state.position.y + 0.24f,
                state.position.z,
            };
            const Vector2 screen = GetWorldToScreen(labelAnchor, camera);

            if (screen.x < 0.0f || screen.x > static_cast<float>(options.width) ||
                screen.y < 0.0f || screen.y > static_cast<float>(options.height)) {
                continue;
            }

            char text[192];
            std::snprintf(text,
                          sizeof(text),
                          "id:%d label:%d (%.2f, %.2f, %.2f)",
                          state.id,
                          state.label,
                          state.position.x,
                          state.position.y,
                          state.position.z);

            Color textColor = colorForObjectId(state.id);
            textColor.a = static_cast<unsigned char>(std::round(fade * 255.0f));
            DrawText(text,
                     static_cast<int>(screen.x) + 8,
                     static_cast<int>(screen.y) - 8,
                     14,
                     textColor);
        }

        DrawRectangle(10, 10, 760, 220, Fade(BLACK, 0.55f));
        DrawRectangleLines(10, 10, 760, 220, Fade(RAYWHITE, 0.35f));

        const double receiveFps = fpsEstimator.fps();
        const size_t activeObjects = objects.size();

        char line0[256];
        char line1[256];
        char line2[256];
        char line3[256];
        char line4[256];
        char line5[256];
        char line6[256];
        char line7[256];

        std::snprintf(line0, sizeof(line0), "render-visual-stream");
        std::snprintf(line1, sizeof(line1), "Upstream: %s", options.ipcEndpoint.c_str());
        if (options.noForward) {
            std::snprintf(line2, sizeof(line2), "Forward: disabled");
        } else {
            std::snprintf(line2,
                          sizeof(line2),
                          "Forward: %s | send-timeout=%d ms recv-timeout=%d ms retries=%d",
                          options.forwardIpcEndpoint.c_str(),
                          options.forwardSendTimeoutMs,
                          options.forwardRecvTimeoutMs,
                          options.forwardRetries);
        }

        if (hasFrame) {
            std::snprintf(line3,
                          sizeof(line3),
                          "Latest frame: %d | ts: %.2f ms | objects in frame: %zu",
                          latestFrameNumber,
                          latestTimestampMs,
                          latestObjectCount);
        } else {
            std::snprintf(line3, sizeof(line3), "Latest frame: (none yet)");
        }

        std::snprintf(line4,
                      sizeof(line4),
                      "Active objects: %zu | Receive FPS: %.2f | Parse failures: %llu | Total parsed: %llu",
                      activeObjects,
                      receiveFps,
                      static_cast<unsigned long long>(stats.parseFailures),
                      static_cast<unsigned long long>(stats.receivedFrames));
        std::snprintf(line5,
                      sizeof(line5),
                      "Forward ok: %llu/%llu | reconnects: %llu",
                      static_cast<unsigned long long>(stats.forwardOk),
                      static_cast<unsigned long long>(stats.forwardAttempts),
                      static_cast<unsigned long long>(stats.forwardReconnects));
        std::snprintf(line6,
                      sizeof(line6),
                      "Forward send fail: %llu (timeout: %llu, error: %llu)",
                      static_cast<unsigned long long>(stats.forwardSendFail),
                      static_cast<unsigned long long>(stats.forwardSendTimeout),
                      static_cast<unsigned long long>(stats.forwardSendError));
        std::snprintf(line7,
                      sizeof(line7),
                      "Forward ack fail: %llu (timeout: %llu, error: %llu, bad-ack: %llu)",
                      static_cast<unsigned long long>(stats.forwardAckFail),
                      static_cast<unsigned long long>(stats.forwardAckTimeout),
                      static_cast<unsigned long long>(stats.forwardAckError),
                      static_cast<unsigned long long>(stats.forwardAckUnexpected));

        DrawText(line0, 20, 18, 20, RAYWHITE);
        DrawText(line1, 20, 43, 16, LIGHTGRAY);
        DrawText(line2, 20, 64, 16, LIGHTGRAY);
        DrawText(line3, 20, 85, 16, LIGHTGRAY);
        DrawText(line4, 20, 106, 16, LIGHTGRAY);
        DrawText(line5, 20, 127, 16, LIGHTGRAY);
        DrawText(line6, 20, 148, 16, LIGHTGRAY);
        DrawText(line7, 20, 169, 16, LIGHTGRAY);
        DrawText("Controls: hold RMB/MMB to move camera | R to reset", 20, 191, 14, GRAY);

        EndDrawing();
        ++renderFrameCounter;

        if (renderFrameCounter % 300 == 0) {
            std::cout << "frames=" << stats.receivedFrames
                      << " active=" << objects.size()
                      << " recv_fps=" << receiveFps
                      << " parse_fail=" << stats.parseFailures
                      << " forward_ok=" << stats.forwardOk
                      << "/" << stats.forwardAttempts
                      << " forward_policy={send_to=" << options.forwardSendTimeoutMs
                      << ",recv_to=" << options.forwardRecvTimeoutMs
                      << ",retries=" << options.forwardRetries << "}"
                      << " forward_send_fail=" << stats.forwardSendFail
                      << " forward_send_timeout=" << stats.forwardSendTimeout
                      << " forward_send_error=" << stats.forwardSendError
                      << " forward_ack_fail=" << stats.forwardAckFail
                      << " forward_ack_timeout=" << stats.forwardAckTimeout
                      << " forward_ack_error=" << stats.forwardAckError
                      << " forward_ack_bad=" << stats.forwardAckUnexpected
                      << " forward_reconnects=" << stats.forwardReconnects
                      << "\n";
        }
    }

    std::cout << "Shutting down render-visual-stream...\n";
    CloseWindow();

    closeSocket(forwardSocket);
    closeSocket(upstreamSocket);
    zmq_ctx_term(zmqContext);
    return 0;
}
