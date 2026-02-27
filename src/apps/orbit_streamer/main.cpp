#include <zmq.h>

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

constexpr const char* DEFAULT_IPC_ENDPOINT = "ipc:///tmp/jv/audio/0.sock";
constexpr double DEFAULT_FPS = 30.0;
constexpr double DEFAULT_RADIUS_METERS = 2.0;
constexpr double DEFAULT_PERIOD_SECONDS = 8.0;
constexpr double DEFAULT_Y_METERS = 0.0;
constexpr double DEFAULT_RADIAL_AMP_METERS = 0.75;
constexpr double DEFAULT_RADIAL_PERIOD_SECONDS = 5.0;
constexpr double DEFAULT_PHASE_OFFSET_DEG = 180.0;
constexpr double MIN_WAVY_RADIUS_METERS = 0.2;
constexpr int DEFAULT_ID_1 = 1;
constexpr int DEFAULT_ID_2 = 2;
constexpr int DEFAULT_LABEL_1 = 0;
constexpr int DEFAULT_LABEL_2 = 0;
constexpr int SOCKET_TIMEOUT_MS = 500;

// Dev-only parser stress setting. Keep at 0 for normal behavior.
constexpr int MALFORMED_FRAME_EVERY_N = 0;

constexpr double kPi = 3.14159265358979323846;

std::atomic<bool> gRunning{true};

void handleSignal(int) {
    gRunning.store(false);
}

enum class MotionMode {
    Orbit,
    Wavy,
    SingleWavy,
};

const char* motionModeToString(MotionMode mode) {
    if (mode == MotionMode::SingleWavy) {
        return "single-wavy";
    }
    if (mode == MotionMode::Wavy) {
        return "wavy";
    }
    return "orbit";
}

bool parseMotionMode(const char* text, MotionMode& out) {
    if (text == nullptr) {
        return false;
    }
    const std::string value(text);
    if (value == "orbit") {
        out = MotionMode::Orbit;
        return true;
    }
    if (value == "wavy") {
        out = MotionMode::Wavy;
        return true;
    }
    if (value == "single-wavy" || value == "single_wavy") {
        out = MotionMode::SingleWavy;
        return true;
    }
    return false;
}

struct CLIOptions {
    std::string ipcEndpoint = DEFAULT_IPC_ENDPOINT;
    double fps = DEFAULT_FPS;
    double radiusMeters = DEFAULT_RADIUS_METERS;
    double periodSec = DEFAULT_PERIOD_SECONDS;
    MotionMode motionMode = MotionMode::Orbit;
    double radialAmpMeters = DEFAULT_RADIAL_AMP_METERS;
    double radialPeriodSec = DEFAULT_RADIAL_PERIOD_SECONDS;
    double phaseOffsetDeg = DEFAULT_PHASE_OFFSET_DEG;
    double yMeters = DEFAULT_Y_METERS;
    int id1 = DEFAULT_ID_1;
    int id2 = DEFAULT_ID_2;
    int label1 = DEFAULT_LABEL_1;
    int label2 = DEFAULT_LABEL_2;
    bool showHelp = false;
    bool valid = true;
};

struct OrbitObject {
    int id = -1;
    int label = 0;
    double x = 0.0;
    double y = 0.0;
    double z = -1.0;
};

struct OrbitFrame {
    int frameNumber = 0;
    double timestampMs = 0.0;
    std::vector<OrbitObject> objects;
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

void printUsage(const char* executableName) {
    std::cout << "Usage: " << executableName << " [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --ipc <endpoint>      ZeroMQ endpoint (default: ipc:///tmp/jv/audio/0.sock)\n";
    std::cout << "  --fps <value>         Frames per second (default: 30.0)\n";
    std::cout << "  --radius <meters>     Orbit radius in meters (default: 2.0)\n";
    std::cout << "  --period-sec <sec>    Orbit period in seconds (default: 8.0)\n";
    std::cout << "  --motion-mode <name>  Motion mode: orbit|wavy|single-wavy (default: orbit)\n";
    std::cout << "  --radial-amp <meters> Radius modulation amplitude (default: 0.75)\n";
    std::cout << "  --radial-period-sec <sec> Radius modulation period (default: 5.0)\n";
    std::cout << "  --phase-offset-deg <deg>  Object 2 radial phase offset for wavy (default: 180.0)\n";
    std::cout << "  --y <meters>          Constant Y height (default: 0.0)\n";
    std::cout << "  --id1 <int>           Object 1 ID (default: 1)\n";
    std::cout << "  --id2 <int>           Object 2 ID (default: 2)\n";
    std::cout << "  --label1 <int>        Object 1 label (default: 0)\n";
    std::cout << "  --label2 <int>        Object 2 label (default: 0)\n";
    std::cout << "  --help, -h            Show this help message\n";
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
    if (end == text || *end != '\0') {
        return false;
    }
    if (value < static_cast<long>(INT32_MIN) || value > static_cast<long>(INT32_MAX)) {
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
            }
            continue;
        }
        if (arg == "--fps") {
            const char* value = requireValue("--fps");
            if (value != nullptr && !parseDouble(value, options.fps)) {
                std::cerr << "Invalid --fps value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--radius") {
            const char* value = requireValue("--radius");
            if (value != nullptr && !parseDouble(value, options.radiusMeters)) {
                std::cerr << "Invalid --radius value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--period-sec") {
            const char* value = requireValue("--period-sec");
            if (value != nullptr && !parseDouble(value, options.periodSec)) {
                std::cerr << "Invalid --period-sec value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--motion-mode") {
            const char* value = requireValue("--motion-mode");
            MotionMode parsedMode = MotionMode::Orbit;
            if (value != nullptr && !parseMotionMode(value, parsedMode)) {
                std::cerr << "Invalid --motion-mode value: " << value
                          << " (expected orbit|wavy|single-wavy)\n";
                options.valid = false;
            } else if (value != nullptr) {
                options.motionMode = parsedMode;
            }
            continue;
        }
        if (arg == "--radial-amp") {
            const char* value = requireValue("--radial-amp");
            if (value != nullptr && !parseDouble(value, options.radialAmpMeters)) {
                std::cerr << "Invalid --radial-amp value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--radial-period-sec") {
            const char* value = requireValue("--radial-period-sec");
            if (value != nullptr && !parseDouble(value, options.radialPeriodSec)) {
                std::cerr << "Invalid --radial-period-sec value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--phase-offset-deg") {
            const char* value = requireValue("--phase-offset-deg");
            if (value != nullptr && !parseDouble(value, options.phaseOffsetDeg)) {
                std::cerr << "Invalid --phase-offset-deg value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--y") {
            const char* value = requireValue("--y");
            if (value != nullptr && !parseDouble(value, options.yMeters)) {
                std::cerr << "Invalid --y value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--id1") {
            const char* value = requireValue("--id1");
            if (value != nullptr && !parseInt(value, options.id1)) {
                std::cerr << "Invalid --id1 value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--id2") {
            const char* value = requireValue("--id2");
            if (value != nullptr && !parseInt(value, options.id2)) {
                std::cerr << "Invalid --id2 value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--label1") {
            const char* value = requireValue("--label1");
            if (value != nullptr && !parseInt(value, options.label1)) {
                std::cerr << "Invalid --label1 value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }
        if (arg == "--label2") {
            const char* value = requireValue("--label2");
            if (value != nullptr && !parseInt(value, options.label2)) {
                std::cerr << "Invalid --label2 value: " << value << "\n";
                options.valid = false;
            }
            continue;
        }

        std::cerr << "Unknown option: " << arg << "\n";
        options.valid = false;
    }

    if (options.fps <= 0.0) {
        std::cerr << "--fps must be > 0\n";
        options.valid = false;
    }
    if (options.radiusMeters <= 0.0) {
        std::cerr << "--radius must be > 0\n";
        options.valid = false;
    }
    if (options.periodSec <= 0.0) {
        std::cerr << "--period-sec must be > 0\n";
        options.valid = false;
    }
    if (options.radialAmpMeters < 0.0) {
        std::cerr << "--radial-amp must be >= 0\n";
        options.valid = false;
    }
    if (options.radialPeriodSec <= 0.0) {
        std::cerr << "--radial-period-sec must be > 0\n";
        options.valid = false;
    }
    if ((options.motionMode == MotionMode::Wavy || options.motionMode == MotionMode::SingleWavy) &&
        options.radiusMeters - options.radialAmpMeters < MIN_WAVY_RADIUS_METERS) {
        std::cerr << "--radius - --radial-amp must be >= " << MIN_WAVY_RADIUS_METERS
                  << " when --motion-mode is wavy or single-wavy\n";
        options.valid = false;
    }

    return options;
}

template <typename T>
void appendPod(std::vector<uint8_t>& out, const T& value) {
    const size_t oldSize = out.size();
    out.resize(oldSize + sizeof(T));
    std::memcpy(out.data() + oldSize, &value, sizeof(T));
}

std::vector<uint8_t> serializeFrame(const OrbitFrame& frame) {
    std::vector<uint8_t> payload;
    const size_t perObjectBytes =
        1 + sizeof(int32_t) + sizeof(int32_t) + 3 * sizeof(double);
    payload.reserve(sizeof(int32_t) + sizeof(double) + 1 + sizeof(int32_t) +
                    frame.objects.size() * perObjectBytes + 1);

    appendPod<int32_t>(payload, static_cast<int32_t>(frame.frameNumber));
    appendPod<double>(payload, frame.timestampMs);
    payload.push_back(static_cast<uint8_t>('^'));
    appendPod<int32_t>(payload, static_cast<int32_t>(frame.objects.size()));

    for (size_t i = 0; i < frame.objects.size(); ++i) {
        const OrbitObject& object = frame.objects[i];
        payload.push_back(static_cast<uint8_t>('|'));
        appendPod<int32_t>(payload, static_cast<int32_t>(object.id));
        appendPod<int32_t>(payload, static_cast<int32_t>(object.label));
        appendPod<double>(payload, object.x);
        appendPod<double>(payload, object.y);
        appendPod<double>(payload, object.z);
    }

    payload.push_back(static_cast<uint8_t>('^'));
    return payload;
}

void maybeInjectMalformedFrame(std::vector<uint8_t>& payload, int frameNumber) {
    if (MALFORMED_FRAME_EVERY_N <= 0) {
        return;
    }
    if (frameNumber <= 0 || frameNumber % MALFORMED_FRAME_EVERY_N != 0) {
        return;
    }

    const size_t markerOffset = sizeof(int32_t) + sizeof(double);
    if (payload.size() > markerOffset) {
        payload[markerOffset] = static_cast<uint8_t>('!');
        std::cerr << "Injected malformed frame for parser sanity test at frame "
                  << frameNumber << "\n";
    }
}

void closeSocket(void*& socket) {
    if (socket != nullptr) {
        zmq_close(socket);
        socket = nullptr;
    }
}

void applySocketOption(void* socket, int option, int value, const char* optionName) {
    if (zmq_setsockopt(socket, option, &value, sizeof(value)) != 0) {
        std::cerr << "Warning: failed to set " << optionName << ": "
                  << zmq_strerror(zmq_errno()) << "\n";
    }
}

void* createReqSocket(void* context, const std::string& endpoint) {
    void* socket = zmq_socket(context, ZMQ_REQ);
    if (socket == nullptr) {
        std::cerr << "Error: failed to create ZMQ_REQ socket: "
                  << zmq_strerror(zmq_errno()) << "\n";
        return nullptr;
    }

    const int linger = 0;
    const int timeout = SOCKET_TIMEOUT_MS;
    applySocketOption(socket, ZMQ_LINGER, linger, "ZMQ_LINGER");
    applySocketOption(socket, ZMQ_SNDTIMEO, timeout, "ZMQ_SNDTIMEO");
    applySocketOption(socket, ZMQ_RCVTIMEO, timeout, "ZMQ_RCVTIMEO");

    if (zmq_connect(socket, endpoint.c_str()) != 0) {
        std::cerr << "Error: failed to connect " << endpoint << ": "
                  << zmq_strerror(zmq_errno()) << "\n";
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
                result.status = SendStatus::Error;
                result.errnum = zmq_errno();
                return result;
            }
        }

        const int sentBytes = zmq_send(socket, payload.data(), payload.size(), 0);
        if (sentBytes == -1) {
            result.errnum = zmq_errno();
            if (attempt == 0) {
                result.retried = true;
                closeSocket(socket);
                continue;
            }
            result.status = (result.errnum == EAGAIN || result.errnum == EINTR)
                                ? SendStatus::Timeout
                                : SendStatus::Error;
            return result;
        }

        char ack[8] = {};
        const int recvBytes = zmq_recv(socket, ack, sizeof(ack), 0);
        if (recvBytes == -1) {
            result.errnum = zmq_errno();
            if (attempt == 0) {
                result.retried = true;
                closeSocket(socket);
                continue;
            }
            result.status = (result.errnum == EAGAIN || result.errnum == EINTR)
                                ? SendStatus::Timeout
                                : SendStatus::Error;
            return result;
        }

        if (recvBytes < 1) {
            result.status = SendStatus::UnknownAck;
            if (attempt == 0) {
                result.retried = true;
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

    result.status = SendStatus::Error;
    return result;
}

void printSummary(const Stats& stats) {
    std::cout << "Summary:\n";
    std::cout << "  sent: " << stats.sent << "\n";
    std::cout << "  acked_ok: " << stats.ackedOk << "\n";
    std::cout << "  acked_fail: " << stats.ackedFail << "\n";
    std::cout << "  timeouts: " << stats.timeouts << "\n";
    std::cout << "  other_errors: " << stats.otherErrors << "\n";
    std::cout << "  retries: " << stats.retries << "\n";
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

    std::cout << "Streaming 3D orbit frames to " << options.ipcEndpoint << "\n";
    std::cout << "motion_mode=" << motionModeToString(options.motionMode)
              << " fps=" << options.fps
              << " radius=" << options.radiusMeters
              << " period=" << options.periodSec
              << " y=" << options.yMeters << "\n";
    if (options.motionMode == MotionMode::Wavy || options.motionMode == MotionMode::SingleWavy) {
        std::cout << "wavy radial_amp=" << options.radialAmpMeters
                  << " radial_period=" << options.radialPeriodSec
                  << " phase_offset_deg=" << options.phaseOffsetDeg << "\n";
    }
    if (MALFORMED_FRAME_EVERY_N > 0) {
        std::cout << "Malformed frame injection enabled every "
                  << MALFORMED_FRAME_EVERY_N << " frames\n";
    }
    std::cout << "Press Ctrl+C to stop.\n";

    const double omega = (2.0 * kPi) / options.periodSec;
    const double radialOmega = (2.0 * kPi) / options.radialPeriodSec;
    const double phaseOffsetRad = options.phaseOffsetDeg * (kPi / 180.0);
    const std::chrono::duration<double> frameIntervalSeconds(1.0 / options.fps);
    const std::chrono::steady_clock::duration frameInterval =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(frameIntervalSeconds);

    auto nextTick = std::chrono::steady_clock::now();
    int frameNumber = 0;
    Stats stats;

    while (gRunning.load()) {
        const double t = static_cast<double>(frameNumber) / options.fps;
        const double theta = omega * t;

        OrbitFrame frame;
        frame.frameNumber = frameNumber;
        frame.timestampMs = t * 1000.0;
        frame.objects.clear();

        const double angleClockwise = -theta;
        const double angleCounterClockwise = theta;
        double radius1 = options.radiusMeters;
        double radius2 = options.radiusMeters;
        if (options.motionMode == MotionMode::Wavy ||
            options.motionMode == MotionMode::SingleWavy) {
            const double radialTheta = radialOmega * t;
            radius1 = options.radiusMeters + options.radialAmpMeters * std::cos(radialTheta);
            if (options.motionMode == MotionMode::Wavy) {
                radius2 = options.radiusMeters +
                          options.radialAmpMeters * std::cos(radialTheta + phaseOffsetRad);
            }
        }

        OrbitObject object1;
        object1.id = options.id1;
        object1.label = options.label1;
        object1.x = radius1 * std::sin(angleClockwise);
        object1.y = options.yMeters;
        object1.z = -radius1 * std::cos(angleClockwise);
        frame.objects.push_back(object1);

        if (options.motionMode != MotionMode::SingleWavy) {
            OrbitObject object2;
            object2.id = options.id2;
            object2.label = options.label2;
            object2.x = radius2 * std::sin(angleCounterClockwise);
            object2.y = options.yMeters;
            object2.z = -radius2 * std::cos(angleCounterClockwise);
            frame.objects.push_back(object2);
        }

        std::vector<uint8_t> payload = serializeFrame(frame);
        maybeInjectMalformedFrame(payload, frameNumber);

        ++stats.sent;
        SendResult result = sendFrameWithSingleRetry(socket, context, options.ipcEndpoint, payload);
        if (result.retried) {
            ++stats.retries;
        }

        if (result.status == SendStatus::Ok) {
            ++stats.ackedOk;
        } else if (result.status == SendStatus::ParseFailAck) {
            ++stats.ackedFail;
            std::cerr << "Receiver parse-fail ACK for frame " << frameNumber << "\n";
        } else if (result.status == SendStatus::Timeout) {
            ++stats.timeouts;
            std::cerr << "Timeout while sending frame " << frameNumber
                      << " (" << zmq_strerror(result.errnum) << ")\n";
        } else if (result.status == SendStatus::UnknownAck) {
            ++stats.otherErrors;
            std::cerr << "Unknown ACK '" << result.ack << "' for frame " << frameNumber << "\n";
        } else {
            ++stats.otherErrors;
            std::cerr << "Socket error while sending frame " << frameNumber;
            if (result.errnum != 0) {
                std::cerr << " (" << zmq_strerror(result.errnum) << ")";
            }
            std::cerr << "\n";
        }

        if (frameNumber % 30 == 0) {
            std::cout << "frame=" << frameNumber
                      << " t_ms=" << frame.timestampMs;
            if (!frame.objects.empty()) {
                std::cout << " obj1=(" << frame.objects[0].x << "," << frame.objects[0].y << ","
                          << frame.objects[0].z << ")";
            }
            if (frame.objects.size() > 1) {
                std::cout << " obj2=(" << frame.objects[1].x << "," << frame.objects[1].y << ","
                          << frame.objects[1].z << ")";
            }
            if (options.motionMode == MotionMode::Wavy) {
                std::cout << " r1=" << radius1 << " r2=" << radius2;
            } else if (options.motionMode == MotionMode::SingleWavy) {
                std::cout << " r1=" << radius1;
            }
            std::cout
                      << " stats{ok=" << stats.ackedOk
                      << ", fail=" << stats.ackedFail
                      << ", timeout=" << stats.timeouts
                      << ", err=" << stats.otherErrors
                      << ", retry=" << stats.retries
                      << "}\n";
        }

        ++frameNumber;
        nextTick += frameInterval;

        const auto now = std::chrono::steady_clock::now();
        if (now < nextTick) {
            std::this_thread::sleep_until(nextTick);
        } else if (now - nextTick > frameInterval * 5) {
            nextTick = now;
        }
    }

    std::cout << "Shutting down orbit streamer...\n";
    printSummary(stats);

    closeSocket(socket);
    zmq_ctx_term(context);
    return 0;
}
