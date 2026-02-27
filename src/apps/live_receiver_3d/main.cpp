#include <phonon.h>
#include <portaudio.h>
#include <zmq.h>

#include <fmt/format.h>
#include <jsa/core/resource_locator.hpp>
#include <jsa/core/wav_io.hpp>
#include <jsa/protocol/frame_parser.hpp>
#include <jsa/tracking/tracker_3d.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr int FALLBACK_SAMPLE_RATE = 44100;
constexpr int MIN_SAMPLE_RATE = 8000;
constexpr int FRAME_SIZE = 256;
constexpr float DEFAULT_FRAME_INTERVAL_MS = 33.333333f;
constexpr float DEFAULT_NO_FRAME_FADE_MS = 50.0f;
constexpr int DEFAULT_STREAM_TIMEOUT_MS = 34;
constexpr float DEFAULT_STALE_FRAME_DROP_MS = 100.0f;
constexpr float DEFAULT_MAX_INTERP_WINDOW_MS = 50.0f;
constexpr float DEFAULT_HOLD_LAST_POSITION_MS = 0.0f;
constexpr size_t HEAVY_OBJECT_WARNING_THRESHOLD = 64;

constexpr float DEFAULT_FEEDBACK_RATE_HZ = 6.0f;
constexpr float DEFAULT_FEEDBACK_DUTY_CYCLE = 0.30f;
constexpr float DEFAULT_TONE_MIN_GAP_MS = 120.0f;
constexpr float DEFAULT_MASTER_GAIN = 0.85f;
constexpr float LIMITER_THRESHOLD = 0.98f;
// C5 root keeps notes in a clearer localization-friendly band than lower octaves.
constexpr float DEFAULT_TONE_BASE_FREQ_HZ = 523.251130f;
constexpr float TONE_FUNDAMENTAL_WEIGHT = 0.78f;
constexpr float TONE_SECOND_HARMONIC_WEIGHT = 0.17f;
constexpr float TONE_THIRD_HARMONIC_WEIGHT = 0.05f;
constexpr float TONE_BRIGHTNESS_DECAY_COEFF = 4.0f;
constexpr float TONE_MAX_NYQUIST_RATIO = 0.45f;
constexpr float TONE_MIN_FREQUENCY_HZ = 20.0f;
constexpr float TWO_PI = 6.28318530717958647692f;
constexpr std::array<float, 9> PENTATONIC_SEMITONES = {
    0.0f, 2.0f, 4.0f, 5.0f, 7.0f, 9.0f, 10.0f, 11.0f, 12.0f};

constexpr const char* DEFAULT_IPC_ENDPOINT = "ipc:///tmp/jv/audio/0.sock";
constexpr const char* DEFAULT_HRTF_SOFA = "D2_HRIR_SOFA/D2_44K_16bit_256tap_FIR_SOFA.sofa";
constexpr const char* DEFAULT_SONG_A_FILE = "lucky.wav";
constexpr const char* DEFAULT_SONG_B_FILE = "september.wav";

const IPLVector3 LISTENER_POSITION = {0.0f, 0.0f, 0.0f};

std::atomic<bool> gRunning{true};

void handleSignal(int) {
    gRunning.store(false);
}

struct WarningLimiter {
    std::chrono::steady_clock::time_point lastWarning =
        std::chrono::steady_clock::time_point::min();

    bool shouldLog() {
        const auto now = std::chrono::steady_clock::now();
        if (now - lastWarning > std::chrono::seconds(1)) {
            lastWarning = now;
            return true;
        }
        return false;
    }
};

WarningLimiter gParseWarningLimiter;
WarningLimiter gLoadWarningLimiter;
WarningLimiter gUnderflowWarningLimiter;
WarningLimiter gNoteOverflowWarningLimiter;
WarningLimiter gTimingStatsLimiter;
WarningLimiter gStaleDropWarningLimiter;

enum class AudioSourceMode {
    Tones,
    Songs
};

enum class ToneScheduleMode {
    Hash,
    RoundRobin
};

std::unordered_set<int> gLoggedSongAssignments;

struct ObjectNoteAssignment {
    float semitone = 0.0f;
    bool isNew = false;
    bool usedOverflowRepeat = false;
};

class ObjectNoteAllocator {
public:
    ObjectNoteAllocator() {
        noteOwner.fill(-1);
    }

    ObjectNoteAssignment assignNote(int objectId) {
        const auto existing = objectToNoteIndex.find(objectId);
        if (existing != objectToNoteIndex.end()) {
            return {PENTATONIC_SEMITONES[existing->second], false, false};
        }

        const size_t poolSize = PENTATONIC_SEMITONES.size();
        const uint32_t h = static_cast<uint32_t>(std::hash<int>{}(objectId));
        const size_t start = static_cast<size_t>(h % static_cast<uint32_t>(poolSize));

        for (size_t probe = 0; probe < poolSize; ++probe) {
            const size_t candidate = (start + probe) % poolSize;
            if (noteOwner[candidate] == -1) {
                noteOwner[candidate] = objectId;
                objectToNoteIndex.emplace(objectId, candidate);
                return {PENTATONIC_SEMITONES[candidate], true, false};
            }
        }

        objectToNoteIndex.emplace(objectId, start);
        return {PENTATONIC_SEMITONES[start], true, true};
    }

private:
    std::unordered_map<int, size_t> objectToNoteIndex;
    std::array<int, PENTATONIC_SEMITONES.size()> noteOwner{};
};

ObjectNoteAllocator gObjectNoteAllocator;

struct CLIOptions {
    std::string ipcEndpoint = DEFAULT_IPC_ENDPOINT;
    std::string assetsRoot;
    bool useDefaultHRTF = true;
    int deviceIndex = -1;
    int sampleRate = 0;
    float feedbackRateHz = DEFAULT_FEEDBACK_RATE_HZ;
    float feedbackDutyCycle = DEFAULT_FEEDBACK_DUTY_CYCLE;
    ToneScheduleMode toneScheduleMode = ToneScheduleMode::RoundRobin;
    float toneMinGapMs = DEFAULT_TONE_MIN_GAP_MS;
    float masterGain = DEFAULT_MASTER_GAIN;
    float noFrameFadeMs = DEFAULT_NO_FRAME_FADE_MS;
    int streamTimeoutMs = DEFAULT_STREAM_TIMEOUT_MS;
    float staleFrameDropMs = DEFAULT_STALE_FRAME_DROP_MS;
    float maxInterpWindowMs = DEFAULT_MAX_INTERP_WINDOW_MS;
    float holdLastPositionMs = DEFAULT_HOLD_LAST_POSITION_MS;
    AudioSourceMode sourceMode = AudioSourceMode::Tones;
    std::string songAPath = DEFAULT_SONG_A_FILE;
    std::string songBPath = DEFAULT_SONG_B_FILE;
    bool showHelp = false;
    bool valid = true;
};

struct RuntimeAudioConfig {
    int sampleRate = FALLBACK_SAMPLE_RATE;
    float feedbackRateHz = DEFAULT_FEEDBACK_RATE_HZ;
    float feedbackDutyCycle = DEFAULT_FEEDBACK_DUTY_CYCLE;
    ToneScheduleMode toneScheduleMode = ToneScheduleMode::RoundRobin;
    float toneMinGapMs = DEFAULT_TONE_MIN_GAP_MS;
    float masterGain = DEFAULT_MASTER_GAIN;
    float limiterThreshold = LIMITER_THRESHOLD;
    float noFrameFadeMs = DEFAULT_NO_FRAME_FADE_MS;
    int streamTimeoutMs = DEFAULT_STREAM_TIMEOUT_MS;
    float staleFrameDropMs = DEFAULT_STALE_FRAME_DROP_MS;
    float maxInterpWindowMs = DEFAULT_MAX_INTERP_WINDOW_MS;
    float holdLastPositionMs = DEFAULT_HOLD_LAST_POSITION_MS;
    AudioSourceMode sourceMode = AudioSourceMode::Tones;
    std::string songAPath = DEFAULT_SONG_A_FILE;
    std::string songBPath = DEFAULT_SONG_B_FILE;
};

struct WAVFile {
    uint32_t sampleRate = 0;
    uint16_t numChannels = 0;
    uint16_t bitsPerSample = 0;
    std::vector<float> samples;
};

struct SongBank {
    WAVFile songA;
    WAVFile songB;

    const WAVFile* songForIndex(int songIndex) const {
        if (songIndex == 0) {
            return &songA;
        }
        if (songIndex == 1) {
            return &songB;
        }
        return nullptr;
    }
};

using SocketObject3D = jsa::protocol::SocketObject3D;
using SocketFrame3D = jsa::protocol::SocketFrame3D;

struct ObjectAudioParams {
    float frequencyHz = DEFAULT_TONE_BASE_FREQ_HZ;
    float beatOffset = 0.0f;
    float beatCycleSec = 1.0f / DEFAULT_FEEDBACK_RATE_HZ;
    float beatDurationSec = (1.0f / DEFAULT_FEEDBACK_RATE_HZ) * DEFAULT_FEEDBACK_DUTY_CYCLE;

    static ObjectAudioParams fromObjectId(int objectId,
                                          float feedbackRateHz,
                                          float feedbackDutyCycle) {
        ObjectAudioParams params;
        const ObjectNoteAssignment noteAssignment = gObjectNoteAllocator.assignNote(objectId);
        params.frequencyHz =
            DEFAULT_TONE_BASE_FREQ_HZ * std::pow(2.0f, noteAssignment.semitone / 12.0f);
        if (noteAssignment.isNew) {
            std::cout << fmt::format("Assigned object {} note {:.1f} st ({:.2f} Hz)\n",
                                     objectId,
                                     noteAssignment.semitone,
                                     params.frequencyHz);
            if (noteAssignment.usedOverflowRepeat &&
                gNoteOverflowWarningLimiter.shouldLog()) {
                std::cerr << fmt::format(
                    "Warning: note pool exhausted ({} notes). Reusing notes for additional objects.\n",
                    PENTATONIC_SEMITONES.size());
            }
        }

        const uint32_t h = static_cast<uint32_t>(std::hash<int>{}(objectId));
        params.beatOffset = static_cast<float>(h % 1000u) / 1000.0f;
        const float safeRate = std::max(0.001f, feedbackRateHz);
        const float safeDuty = std::max(0.001f, std::min(1.0f, feedbackDutyCycle));
        params.beatCycleSec = 1.0f / safeRate;
        params.beatDurationSec = params.beatCycleSec * safeDuty;
        return params;
    }
};

struct ToneScheduleEntry {
    int objectId = -1;
    ObjectAudioParams params{};
};

struct SourceVoiceState {
    int id = -1;
    int label = -1;
    IPLDirectEffect directEffect = nullptr;
    IPLBinauralEffect binauralEffect = nullptr;
    ObjectAudioParams audioParams{};
    int songIndex = 0;
    double songCursorSamples = 0.0;
    bool wasActiveInLatestFrame = false;
};

using ActiveObjectSnapshot = jsa::tracking::ActiveObjectSnapshot;
using ObjectTracker3D = jsa::tracking::Tracker3D;

struct SteamAudioResources {
    IPLContext context = nullptr;
    IPLHRTF hrtf = nullptr;
    IPLAudioSettings audioSettings{};
    IPLDistanceAttenuationModel distanceModel{};
    IPLAirAbsorptionModel airAbsorptionModel{};
    IPLAudioBuffer inputBuffer{};
    IPLAudioBuffer outputBuffer{};
    IPLAudioBuffer mixBuffer{};

    ~SteamAudioResources() {
        release();
    }

    void release();
};

void printUsage(const char* executableName) {
    std::cout << "Usage: " << executableName << " [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --ipc <endpoint>            ZeroMQ endpoint (default: ipc:///tmp/jv/audio/0.sock)\n";
    std::cout << "  --assets-root <path>        Asset root override (highest priority)\n";
    std::cout << "  --hrtf <default|custom>     HRTF type (default: default)\n";
    std::cout << "  --device-index <index>      PortAudio output device index (default: -1)\n";
    std::cout << "  --sample-rate <hz>          Output/engine sample rate (0=auto, default: 0)\n";
    std::cout << "  --feedback-rate-hz <hz>     Beep pulse rate (default: 6.0)\n";
    std::cout << "  --feedback-duty-cycle <0-1> Pulse duty cycle (default: 0.30)\n";
    std::cout << "  --tone-schedule <hash|round-robin> Tone scheduling strategy (default: round-robin)\n";
    std::cout << "  --tone-min-gap-ms <ms>      Minimum gap target between object beeps (default: 120.0)\n";
    std::cout << "  --master-gain <0-1>         Post-mix gain (default: 0.85)\n";
    std::cout << "  --no-frame-fade-ms <ms>     Fade-out duration when no frame arrives (default: 50.0)\n";
    std::cout << "  --stream-timeout-ms <ms>    ZeroMQ receive timeout and max render tick size (default: 34)\n";
    std::cout << "  --stale-frame-drop-ms <ms>  Drop frames older than this staleness budget (default: 100.0)\n";
    std::cout << "  --max-interp-window-ms <ms> Max interpolation window per render tick (default: 50.0)\n";
    std::cout << "  --hold-last-position-ms <ms> Hold object position before fade on missing frames (default: 0.0)\n";
    std::cout << "  --source-mode <tones|songs> Source generator (default: tones)\n";
    std::cout << "  --song-a <wav>              Song A path for songs mode (default: lucky.wav)\n";
    std::cout << "  --song-b <wav>              Song B path for songs mode (default: september.wav)\n";
    std::cout << "  --help, -h                  Show this help message\n";
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

bool parseSourceModeArg(std::string_view text, AudioSourceMode& outMode) {
    if (text == "tones") {
        outMode = AudioSourceMode::Tones;
        return true;
    }
    if (text == "songs") {
        outMode = AudioSourceMode::Songs;
        return true;
    }
    return false;
}

bool parseToneScheduleModeArg(std::string_view text, ToneScheduleMode& outMode) {
    if (text == "hash") {
        outMode = ToneScheduleMode::Hash;
        return true;
    }
    if (text == "round-robin") {
        outMode = ToneScheduleMode::RoundRobin;
        return true;
    }
    return false;
}

CLIOptions parseCommandLine(int argc, char* argv[]) {
    CLIOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            options.showHelp = true;
            continue;
        }

        auto requireValue = [&](const char* optionName) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << fmt::format("Missing value for option: {}\n", optionName);
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
        if (arg == "--assets-root") {
            const char* value = requireValue("--assets-root");
            if (value != nullptr) {
                options.assetsRoot = value;
            }
            continue;
        }
        if (arg == "--hrtf") {
            const char* value = requireValue("--hrtf");
            if (value == nullptr) {
                continue;
            }

            const std::string_view hrtfType(value);
            if (hrtfType == "default") {
                options.useDefaultHRTF = true;
            } else if (hrtfType == "custom") {
                options.useDefaultHRTF = false;
            } else {
                std::cerr << fmt::format("Invalid --hrtf value: {}\n", hrtfType);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--device-index") {
            const char* value = requireValue("--device-index");
            if (value != nullptr && !parseIntArg(value, options.deviceIndex)) {
                std::cerr << fmt::format("Invalid --device-index value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--sample-rate") {
            const char* value = requireValue("--sample-rate");
            if (value != nullptr && !parseIntArg(value, options.sampleRate)) {
                std::cerr << fmt::format("Invalid --sample-rate value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--feedback-rate-hz") {
            const char* value = requireValue("--feedback-rate-hz");
            if (value != nullptr && !parseFloatArg(value, options.feedbackRateHz)) {
                std::cerr << fmt::format("Invalid --feedback-rate-hz value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--feedback-duty-cycle") {
            const char* value = requireValue("--feedback-duty-cycle");
            if (value != nullptr && !parseFloatArg(value, options.feedbackDutyCycle)) {
                std::cerr << fmt::format("Invalid --feedback-duty-cycle value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--tone-schedule") {
            const char* value = requireValue("--tone-schedule");
            ToneScheduleMode parsedSchedule = options.toneScheduleMode;
            if (value == nullptr || !parseToneScheduleModeArg(value, parsedSchedule)) {
                std::cerr << fmt::format(
                    "Invalid --tone-schedule value: {} (expected hash or round-robin)\n",
                    value != nullptr ? value : "<null>");
                options.valid = false;
            } else {
                options.toneScheduleMode = parsedSchedule;
            }
            continue;
        }
        if (arg == "--tone-min-gap-ms") {
            const char* value = requireValue("--tone-min-gap-ms");
            if (value != nullptr && !parseFloatArg(value, options.toneMinGapMs)) {
                std::cerr << fmt::format("Invalid --tone-min-gap-ms value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--master-gain") {
            const char* value = requireValue("--master-gain");
            if (value != nullptr && !parseFloatArg(value, options.masterGain)) {
                std::cerr << fmt::format("Invalid --master-gain value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--no-frame-fade-ms") {
            const char* value = requireValue("--no-frame-fade-ms");
            if (value != nullptr && !parseFloatArg(value, options.noFrameFadeMs)) {
                std::cerr << fmt::format("Invalid --no-frame-fade-ms value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--stream-timeout-ms") {
            const char* value = requireValue("--stream-timeout-ms");
            if (value != nullptr && !parseIntArg(value, options.streamTimeoutMs)) {
                std::cerr << fmt::format("Invalid --stream-timeout-ms value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--stale-frame-drop-ms") {
            const char* value = requireValue("--stale-frame-drop-ms");
            if (value != nullptr && !parseFloatArg(value, options.staleFrameDropMs)) {
                std::cerr << fmt::format("Invalid --stale-frame-drop-ms value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--max-interp-window-ms") {
            const char* value = requireValue("--max-interp-window-ms");
            if (value != nullptr && !parseFloatArg(value, options.maxInterpWindowMs)) {
                std::cerr << fmt::format("Invalid --max-interp-window-ms value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--hold-last-position-ms") {
            const char* value = requireValue("--hold-last-position-ms");
            if (value != nullptr && !parseFloatArg(value, options.holdLastPositionMs)) {
                std::cerr << fmt::format("Invalid --hold-last-position-ms value: {}\n", value);
                options.valid = false;
            }
            continue;
        }
        if (arg == "--source-mode") {
            const char* value = requireValue("--source-mode");
            AudioSourceMode parsedMode = options.sourceMode;
            if (value == nullptr || !parseSourceModeArg(value, parsedMode)) {
                std::cerr << fmt::format(
                    "Invalid --source-mode value: {} (expected tones or songs)\n",
                    value != nullptr ? value : "<null>");
                options.valid = false;
            } else {
                options.sourceMode = parsedMode;
            }
            continue;
        }
        if (arg == "--song-a") {
            const char* value = requireValue("--song-a");
            if (value != nullptr) {
                options.songAPath = value;
            }
            continue;
        }
        if (arg == "--song-b") {
            const char* value = requireValue("--song-b");
            if (value != nullptr) {
                options.songBPath = value;
            }
            continue;
        }

        std::cerr << fmt::format("Unknown option: {}\n", arg);
        options.valid = false;
    }

    if (options.feedbackRateHz <= 0.0f) {
        std::cerr << "--feedback-rate-hz must be > 0\n";
        options.valid = false;
    }
    if (options.feedbackDutyCycle <= 0.0f || options.feedbackDutyCycle > 1.0f) {
        std::cerr << "--feedback-duty-cycle must be in (0, 1]\n";
        options.valid = false;
    }
    if (options.toneMinGapMs <= 0.0f) {
        std::cerr << "--tone-min-gap-ms must be > 0\n";
        options.valid = false;
    }
    if (options.sampleRate != 0 && options.sampleRate < MIN_SAMPLE_RATE) {
        std::cerr << fmt::format("--sample-rate must be 0 or >= {}\n", MIN_SAMPLE_RATE);
        options.valid = false;
    }
    if (options.masterGain <= 0.0f || options.masterGain > 1.0f) {
        std::cerr << "--master-gain must be in (0, 1]\n";
        options.valid = false;
    }
    if (options.noFrameFadeMs <= 0.0f) {
        std::cerr << "--no-frame-fade-ms must be > 0\n";
        options.valid = false;
    }
    if (options.streamTimeoutMs <= 0) {
        std::cerr << "--stream-timeout-ms must be > 0\n";
        options.valid = false;
    }
    if (options.staleFrameDropMs <= 0.0f) {
        std::cerr << "--stale-frame-drop-ms must be > 0\n";
        options.valid = false;
    }
    if (options.maxInterpWindowMs <= 0.0f) {
        std::cerr << "--max-interp-window-ms must be > 0\n";
        options.valid = false;
    }
    if (options.holdLastPositionMs < 0.0f) {
        std::cerr << "--hold-last-position-ms must be >= 0\n";
        options.valid = false;
    }
    if (options.sourceMode == AudioSourceMode::Songs) {
        if (options.songAPath.empty()) {
            std::cerr << "--song-a must not be empty when --source-mode songs\n";
            options.valid = false;
        }
        if (options.songBPath.empty()) {
            std::cerr << "--song-b must not be empty when --source-mode songs\n";
            options.valid = false;
        }
    }

    return options;
}

int songIndexForObjectId(int objectId) {
    const uint32_t h = static_cast<uint32_t>(std::hash<int>{}(objectId));
    return static_cast<int>(h % 2u);
}

const std::string& songPathForIndex(const RuntimeAudioConfig& runtimeConfig, int songIndex) {
    if (songIndex == 0) {
        return runtimeConfig.songAPath;
    }
    return runtimeConfig.songBPath;
}

bool readWAV(const std::string& filename, WAVFile& wav) {
    jsa::core::WavData loaded;
    std::string err;
    if (!jsa::core::loadWavFile(filename, loaded, err)) {
        std::cerr << fmt::format("Error: {}\n", err);
        return false;
    }

    wav.sampleRate = static_cast<uint32_t>(loaded.sampleRate);
    wav.numChannels = static_cast<uint16_t>(loaded.channels);
    wav.bitsPerSample = static_cast<uint16_t>(loaded.bitsPerSample);
    wav.samples = std::move(loaded.samples);

    if (wav.samples.empty()) {
        std::cerr << fmt::format("Error: WAV contains no samples: {}\n", filename);
        return false;
    }

    std::cout << fmt::format(
        "Loaded WAV: {} ({} Hz, {} mono samples)\n", filename, wav.sampleRate, wav.samples.size());
    return true;
}

std::vector<ToneScheduleEntry> buildRoundRobinToneSchedule(
    const std::vector<ActiveObjectSnapshot>& activeObjects,
    const std::unordered_map<int, SourceVoiceState>& voices,
    float feedbackRateHz,
    float feedbackDutyCycle,
    float toneMinGapMs) {
    std::vector<ToneScheduleEntry> scheduledEntries;
    if (activeObjects.empty()) {
        return scheduledEntries;
    }

    struct RankedObject {
        int objectId = -1;
        float distance = 0.0f;
    };

    std::vector<RankedObject> ranked;
    ranked.reserve(activeObjects.size());
    for (const ActiveObjectSnapshot& object : activeObjects) {
        ranked.push_back(
            {object.id,
             std::sqrt(object.position.x * object.position.x +
                       object.position.y * object.position.y +
                       object.position.z * object.position.z)});
    }

    std::sort(ranked.begin(), ranked.end(), [](const RankedObject& a, const RankedObject& b) {
        if (a.distance == b.distance) {
            return a.objectId < b.objectId;
        }
        return a.distance < b.distance;
    });

    const float safeRate = std::max(0.001f, feedbackRateHz);
    const float safeDuty = std::max(0.001f, std::min(1.0f, feedbackDutyCycle));
    const float toneMinGapSec = std::max(0.001f, toneMinGapMs / 1000.0f);
    const float activeCount = static_cast<float>(ranked.size());
    const float baseCycleSec = 1.0f / safeRate;
    const float minCycleSec = activeCount * toneMinGapSec;
    const float cycleSec = std::max(baseCycleSec, minCycleSec);
    const float slotSec = cycleSec / activeCount;
    const float beepSec = std::min(slotSec * safeDuty, slotSec * 0.8f);

    scheduledEntries.reserve(ranked.size());
    for (size_t k = 0; k < ranked.size(); ++k) {
        const int objectId = ranked[k].objectId;
        ObjectAudioParams params;
        params.frequencyHz = DEFAULT_TONE_BASE_FREQ_HZ;
        const auto voiceIt = voices.find(objectId);
        if (voiceIt != voices.end()) {
            params.frequencyHz = voiceIt->second.audioParams.frequencyHz;
        }

        params.beatCycleSec = cycleSec;
        params.beatDurationSec = beepSec;
        params.beatOffset = static_cast<float>(k) / activeCount;
        scheduledEntries.push_back({objectId, params});
    }

    return scheduledEntries;
}

void freeBuffer(IPLContext context, IPLAudioBuffer& buffer) {
    if (context != nullptr && buffer.data != nullptr) {
        iplAudioBufferFree(context, &buffer);
    }
    buffer.data = nullptr;
    buffer.numChannels = 0;
    buffer.numSamples = 0;
}

void SteamAudioResources::release() {
    freeBuffer(context, inputBuffer);
    freeBuffer(context, outputBuffer);
    freeBuffer(context, mixBuffer);

    if (hrtf != nullptr) {
        iplHRTFRelease(&hrtf);
        hrtf = nullptr;
    }

    if (context != nullptr) {
        iplContextRelease(&context);
        context = nullptr;
    }
}

void releaseVoice(SourceVoiceState& voice) {
    if (voice.directEffect != nullptr) {
        iplDirectEffectRelease(&voice.directEffect);
        voice.directEffect = nullptr;
    }
    if (voice.binauralEffect != nullptr) {
        iplBinauralEffectRelease(&voice.binauralEffect);
        voice.binauralEffect = nullptr;
    }
}

void zeroBuffer(IPLAudioBuffer& buffer) {
    if (buffer.data == nullptr) {
        return;
    }
    for (int channel = 0; channel < buffer.numChannels; ++channel) {
        std::fill(buffer.data[channel], buffer.data[channel] + buffer.numSamples, 0.0f);
    }
}

bool allocateBuffer(IPLContext context,
                    int channels,
                    int numSamples,
                    IPLAudioBuffer& buffer,
                    const std::string& name) {
    buffer.numChannels = channels;
    buffer.numSamples = numSamples;
    buffer.data = nullptr;

    const IPLerror error = iplAudioBufferAllocate(context, channels, numSamples, &buffer);
    if (error != IPL_STATUS_SUCCESS || buffer.data == nullptr) {
        std::cerr << fmt::format("Error: failed to allocate {} buffer\n", name);
        return false;
    }
    return true;
}

void getToneSamples(float* output,
                    int numSamples,
                    float startTimeSec,
                    float targetSampleRate,
                    const ObjectAudioParams& params) {
    if (output == nullptr || numSamples <= 0) {
        return;
    }

    if (targetSampleRate <= 0.0f || params.beatCycleSec <= 0.0f || params.beatDurationSec <= 0.0f) {
        std::fill(output, output + numSamples, 0.0f);
        return;
    }

    const float safeSampleRate = std::max(1.0f, targetSampleRate);
    const float slotStart = params.beatOffset * params.beatCycleSec;
    const float slotEnd = slotStart + params.beatDurationSec;
    const float maxFrequencyHz =
        std::max(TONE_MIN_FREQUENCY_HZ, TONE_MAX_NYQUIST_RATIO * safeSampleRate);
    const float frequencyHz =
        std::max(TONE_MIN_FREQUENCY_HZ, std::min(params.frequencyHz, maxFrequencyHz));
    const float attackSec = std::max(1.0f / safeSampleRate, std::min(0.008f, params.beatDurationSec * 0.20f));
    const float releaseSec =
        std::max(1.0f / safeSampleRate, std::min(0.030f, params.beatDurationSec * 0.35f));

    for (int i = 0; i < numSamples; ++i) {
        const float targetTime = startTimeSec + (static_cast<float>(i) / safeSampleRate);
        float cyclePosition = std::fmod(targetTime, params.beatCycleSec);
        if (cyclePosition < 0.0f) {
            cyclePosition += params.beatCycleSec;
        }

        bool inSlot = false;
        if (slotEnd <= params.beatCycleSec) {
            inSlot = (cyclePosition >= slotStart && cyclePosition < slotEnd);
        } else {
            inSlot =
                (cyclePosition >= slotStart || cyclePosition < std::fmod(slotEnd, params.beatCycleSec));
        }

        if (!inSlot) {
            output[i] = 0.0f;
            continue;
        }

        float timeInSlot = cyclePosition - slotStart;
        if (timeInSlot < 0.0f) {
            timeInSlot += params.beatCycleSec;
        }

        if (timeInSlot < 0.0f || timeInSlot >= params.beatDurationSec) {
            output[i] = 0.0f;
            continue;
        }

        const float slotProgress =
            std::max(0.0f, std::min(timeInSlot / params.beatDurationSec, 1.0f));
        float attackEnvelope = 1.0f;
        if (timeInSlot < attackSec) {
            attackEnvelope = timeInSlot / attackSec;
        }

        const float timeRemainingInSlot = params.beatDurationSec - timeInSlot;
        float releaseEnvelope = 1.0f;
        if (timeRemainingInSlot < releaseSec) {
            releaseEnvelope = std::max(0.0f, timeRemainingInSlot / releaseSec);
        }

        const float edgeEnvelope = std::min(attackEnvelope, releaseEnvelope);
        const float bellEnvelope = std::exp(-TONE_BRIGHTNESS_DECAY_COEFF * slotProgress);

        const float basePhase = TWO_PI * frequencyHz * timeInSlot;
        const float tone =
            (TONE_FUNDAMENTAL_WEIGHT * std::sin(basePhase)) +
            (TONE_SECOND_HARMONIC_WEIGHT * std::sin(2.0f * basePhase)) +
            (TONE_THIRD_HARMONIC_WEIGHT * std::sin(3.0f * basePhase));

        output[i] = tone * edgeEnvelope * bellEnvelope;
    }
}

void getLoopedSongSamples(const WAVFile& wav,
                          float* output,
                          int numSamples,
                          float targetSampleRate,
                          double& cursorSamples) {
    if (output == nullptr || numSamples <= 0) {
        return;
    }

    if (wav.samples.empty() || wav.sampleRate == 0 || targetSampleRate <= 0.0f) {
        std::fill(output, output + numSamples, 0.0f);
        return;
    }

    const size_t sourceSizeInt = wav.samples.size();
    const double sourceSize = static_cast<double>(sourceSizeInt);
    const double step = static_cast<double>(wav.sampleRate) / static_cast<double>(targetSampleRate);
    double cursor = cursorSamples;
    if (!std::isfinite(cursor)) {
        cursor = 0.0;
    }
    cursor = std::fmod(cursor, sourceSize);
    if (cursor < 0.0) {
        cursor += sourceSize;
    }

    for (int i = 0; i < numSamples; ++i) {
        const size_t index = static_cast<size_t>(cursor);
        const size_t nextIndex = (index + 1u) % sourceSizeInt;
        const double fraction = cursor - static_cast<double>(index);
        const float sample = wav.samples[index] +
                             (wav.samples[nextIndex] - wav.samples[index]) *
                                 static_cast<float>(fraction);
        output[i] = sample;

        cursor += step;
        if (cursor >= sourceSize) {
            cursor = std::fmod(cursor, sourceSize);
        }
    }

    cursorSamples = cursor;
}

bool createHRTF(bool useDefault,
                const std::string& customHrtfPath,
                SteamAudioResources& resources) {
    IPLHRTFSettings hrtfSettings{};
    hrtfSettings.volume = 1.0f;
    hrtfSettings.normType = IPL_HRTFNORMTYPE_NONE;

    IPLerror error = IPL_STATUS_SUCCESS;
    if (useDefault) {
        hrtfSettings.type = IPL_HRTFTYPE_DEFAULT;
        error = iplHRTFCreate(resources.context,
                              &resources.audioSettings,
                              &hrtfSettings,
                              &resources.hrtf);
    } else {
        hrtfSettings.type = IPL_HRTFTYPE_SOFA;
        hrtfSettings.sofaFileName = customHrtfPath.c_str();
        hrtfSettings.sofaData = nullptr;
        hrtfSettings.sofaDataSize = 0;
        error = iplHRTFCreate(resources.context,
                              &resources.audioSettings,
                              &hrtfSettings,
                              &resources.hrtf);
    }

    if (error != IPL_STATUS_SUCCESS) {
        std::cerr << "Error: failed to create HRTF\n";
        return false;
    }
    return true;
}

bool initializeSteamAudioResources(bool useDefaultHRTF,
                                   int sampleRate,
                                   const std::string& customHrtfPath,
                                   SteamAudioResources& resources) {
    resources.release();

    IPLContextSettings contextSettings{};
    contextSettings.version = STEAMAUDIO_VERSION;
    contextSettings.simdLevel = IPL_SIMDLEVEL_NEON;

    IPLerror error = iplContextCreate(&contextSettings, &resources.context);
    if (error != IPL_STATUS_SUCCESS) {
        std::cerr << "Error: failed to create Steam Audio context\n";
        return false;
    }

    resources.audioSettings.samplingRate = sampleRate;
    resources.audioSettings.frameSize = FRAME_SIZE;

    if (!createHRTF(useDefaultHRTF, customHrtfPath, resources)) {
        resources.release();
        return false;
    }

    resources.distanceModel = {};
    resources.distanceModel.type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
    resources.distanceModel.minDistance = 0.0f;
    resources.distanceModel.callback = nullptr;
    resources.distanceModel.userData = nullptr;
    resources.distanceModel.dirty = IPL_FALSE;

    resources.airAbsorptionModel = {};
    resources.airAbsorptionModel.type = IPL_AIRABSORPTIONTYPE_DEFAULT;
    resources.airAbsorptionModel.callback = nullptr;
    resources.airAbsorptionModel.userData = nullptr;
    resources.airAbsorptionModel.dirty = IPL_FALSE;

    if (!allocateBuffer(resources.context, 1, FRAME_SIZE, resources.inputBuffer, "input") ||
        !allocateBuffer(resources.context, 2, FRAME_SIZE, resources.outputBuffer, "output") ||
        !allocateBuffer(resources.context, 2, FRAME_SIZE, resources.mixBuffer, "mix")) {
        resources.release();
        return false;
    }

    return true;
}

SourceVoiceState* ensureVoiceForObject(int objectId,
                                       int label,
                                       const RuntimeAudioConfig& runtimeConfig,
                                       SteamAudioResources& resources,
                                       std::unordered_map<int, SourceVoiceState>& voices) {
    auto it = voices.find(objectId);
    if (it != voices.end()) {
        it->second.label = label;
        return &it->second;
    }

    SourceVoiceState newVoice;
    newVoice.id = objectId;
    newVoice.label = label;
    if (runtimeConfig.sourceMode == AudioSourceMode::Tones) {
        newVoice.audioParams = ObjectAudioParams::fromObjectId(objectId,
                                                               runtimeConfig.feedbackRateHz,
                                                               runtimeConfig.feedbackDutyCycle);
    }
    newVoice.songIndex = songIndexForObjectId(objectId);
    newVoice.songCursorSamples = 0.0;
    if (runtimeConfig.sourceMode == AudioSourceMode::Songs &&
        gLoggedSongAssignments.insert(objectId).second) {
        std::cout << fmt::format("Assigned object {} song {}\n",
                                 objectId,
                                 songPathForIndex(runtimeConfig, newVoice.songIndex));
    }

    IPLBinauralEffectSettings binauralSettings{};
    binauralSettings.hrtf = resources.hrtf;
    IPLerror error = iplBinauralEffectCreate(resources.context,
                                             &resources.audioSettings,
                                             &binauralSettings,
                                             &newVoice.binauralEffect);
    if (error != IPL_STATUS_SUCCESS) {
        std::cerr << fmt::format("Error: failed to create binaural effect for object {}\n", objectId);
        return nullptr;
    }

    IPLDirectEffectSettings directSettings{};
    directSettings.numChannels = 1;
    error = iplDirectEffectCreate(resources.context,
                                  &resources.audioSettings,
                                  &directSettings,
                                  &newVoice.directEffect);
    if (error != IPL_STATUS_SUCCESS) {
        std::cerr << fmt::format("Error: failed to create direct effect for object {}\n", objectId);
        if (newVoice.binauralEffect != nullptr) {
            iplBinauralEffectRelease(&newVoice.binauralEffect);
            newVoice.binauralEffect = nullptr;
        }
        return nullptr;
    }

    auto insertResult = voices.emplace(objectId, std::move(newVoice));
    if (!insertResult.second) {
        insertResult.first->second.label = label;
        return &insertResult.first->second;
    }

    return &insertResult.first->second;
}

void releaseVoiceForObject(int objectId, std::unordered_map<int, SourceVoiceState>& voices) {
    auto it = voices.find(objectId);
    if (it == voices.end()) {
        return;
    }

    releaseVoice(it->second);
    voices.erase(it);
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

bool initializeOutputStream(int requestedDeviceIndex,
                            int requestedSampleRate,
                            PaStream*& stream,
                            int& actualSampleRate) {
    stream = nullptr;
    actualSampleRate = FALLBACK_SAMPLE_RATE;

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << fmt::format("Error: Pa_Initialize failed: {}\n", Pa_GetErrorText(err));
        return false;
    }

    const int deviceCount = Pa_GetDeviceCount();
    if (deviceCount < 0) {
        std::cerr << "Error: failed to query PortAudio devices\n";
        Pa_Terminate();
        return false;
    }

    int outputDevice = requestedDeviceIndex;
    if (outputDevice < 0) {
        outputDevice = Pa_GetDefaultOutputDevice();
    }
    if (outputDevice == paNoDevice || outputDevice < 0 || outputDevice >= deviceCount) {
        std::cerr << "Error: no valid output device available\n";
        Pa_Terminate();
        return false;
    }

    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(outputDevice);
    if (deviceInfo == nullptr) {
        std::cerr << "Error: failed to query selected output device\n";
        Pa_Terminate();
        return false;
    }

    std::cout << fmt::format("Using PortAudio output device {}: {}\n",
                             outputDevice,
                             deviceInfo->name ? deviceInfo->name : "Unknown");

    int selectedSampleRate = requestedSampleRate;
    const int deviceDefaultSampleRate = static_cast<int>(std::llround(deviceInfo->defaultSampleRate));
    if (selectedSampleRate <= 0) {
        selectedSampleRate = deviceDefaultSampleRate;
    }
    if (selectedSampleRate < MIN_SAMPLE_RATE) {
        selectedSampleRate = FALLBACK_SAMPLE_RATE;
    }

    PaStreamParameters outputParameters{};
    outputParameters.device = outputDevice;
    outputParameters.channelCount = 2;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency = deviceInfo->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(&stream,
                        nullptr,
                        &outputParameters,
                        static_cast<double>(selectedSampleRate),
                        FRAME_SIZE,
                        paClipOff,
                        nullptr,
                        nullptr);
    if (err != paNoError) {
        std::cerr << fmt::format("Error: Pa_OpenStream failed: {}\n", Pa_GetErrorText(err));
        Pa_Terminate();
        return false;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << fmt::format("Error: Pa_StartStream failed: {}\n", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        stream = nullptr;
        Pa_Terminate();
        return false;
    }

    std::cout << fmt::format("Device default sample rate: {} Hz, engine sample rate: {} Hz\n",
                             deviceDefaultSampleRate,
                             selectedSampleRate);
    actualSampleRate = selectedSampleRate;
    return true;
}

void shutdownOutputStream(PaStream*& stream) {
    if (stream != nullptr) {
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        stream = nullptr;
    }
    Pa_Terminate();
}

bool renderAndPlaySamples(const ObjectTracker3D& tracker,
                          int samplesToRender,
                          double interpolationHintMs,
                          double& audioTimeSec,
                          const RuntimeAudioConfig& runtimeConfig,
                          const SongBank* songBank,
                          SteamAudioResources& resources,
                          PaStream* stream,
                          std::unordered_map<int, SourceVoiceState>& voices,
                          double& maxInterpolationWindowUsedMs,
                          uint64_t& outputUnderflowCount) {
    if (samplesToRender <= 0) {
        return true;
    }

    const double safeSampleRate = std::max(1, runtimeConfig.sampleRate);
    const double renderWindowMs =
        (static_cast<double>(samplesToRender) * 1000.0) / safeSampleRate;
    double sanitizedInterpolationHintMs = interpolationHintMs;
    if (!std::isfinite(sanitizedInterpolationHintMs) || sanitizedInterpolationHintMs <= 0.0) {
        sanitizedInterpolationHintMs = renderWindowMs;
    }
    const double effectiveInterpolationWindowMs = std::max(
        1.0,
        std::min({renderWindowMs,
                  static_cast<double>(runtimeConfig.maxInterpWindowMs),
                  sanitizedInterpolationHintMs}));
    int effectiveInterpolationSamples = std::max(
        1,
        static_cast<int>(
            std::llround((effectiveInterpolationWindowMs / 1000.0) * safeSampleRate)));
    effectiveInterpolationSamples = std::min(effectiveInterpolationSamples, samplesToRender);
    const double usedInterpolationWindowMs =
        (static_cast<double>(effectiveInterpolationSamples) * 1000.0) / safeSampleRate;
    maxInterpolationWindowUsedMs =
        std::max(maxInterpolationWindowUsedMs, usedInterpolationWindowMs);

    int rendered = 0;
    std::vector<float> interleaved;
    interleaved.reserve(static_cast<size_t>(FRAME_SIZE) * 2);

    while (rendered < samplesToRender && gRunning.load()) {
        const int chunkSamples = std::min(FRAME_SIZE, samplesToRender - rendered);
        const int interpolationProgressSamples =
            std::min(rendered + chunkSamples, effectiveInterpolationSamples);
        const float interpolationFactor = static_cast<float>(interpolationProgressSamples) /
                                          static_cast<float>(effectiveInterpolationSamples);

        const auto activeObjects = tracker.getInterpolatedActiveObjects(interpolationFactor);
        if (activeObjects.size() > HEAVY_OBJECT_WARNING_THRESHOLD && gLoadWarningLimiter.shouldLog()) {
            std::cerr << fmt::format(
                "Warning: high active object count ({}). Rendering all objects with no cap.\n",
                activeObjects.size());
        }

        zeroBuffer(resources.mixBuffer);
        std::unordered_map<int, ObjectAudioParams> toneParamsByObject;
        if (runtimeConfig.sourceMode == AudioSourceMode::Tones &&
            runtimeConfig.toneScheduleMode == ToneScheduleMode::RoundRobin &&
            !activeObjects.empty()) {
            for (const auto& object : activeObjects) {
                ensureVoiceForObject(object.id, object.label, runtimeConfig, resources, voices);
            }
            const auto scheduledEntries = buildRoundRobinToneSchedule(activeObjects,
                                                                      voices,
                                                                      runtimeConfig.feedbackRateHz,
                                                                      runtimeConfig.feedbackDutyCycle,
                                                                      runtimeConfig.toneMinGapMs);
            toneParamsByObject.reserve(scheduledEntries.size());
            for (const ToneScheduleEntry& entry : scheduledEntries) {
                toneParamsByObject.emplace(entry.objectId, entry.params);
            }
        }

        for (const auto& object : activeObjects) {
            SourceVoiceState* voice =
                ensureVoiceForObject(object.id, object.label, runtimeConfig, resources, voices);
            if (voice == nullptr) {
                continue;
            }

            const bool activeInLatestFrame = tracker.isActiveInLatestFrame(object.id);
            if (runtimeConfig.sourceMode == AudioSourceMode::Songs &&
                activeInLatestFrame &&
                !voice->wasActiveInLatestFrame) {
                voice->songCursorSamples = 0.0;
            }
            voice->wasActiveInLatestFrame = activeInLatestFrame;

            const float distance = std::max(
                0.001f,
                std::sqrt(object.position.x * object.position.x +
                          object.position.y * object.position.y +
                          object.position.z * object.position.z));
            IPLVector3 direction{
                object.position.x / distance,
                object.position.y / distance,
                object.position.z / distance};

            zeroBuffer(resources.inputBuffer);
            if (runtimeConfig.sourceMode == AudioSourceMode::Songs) {
                const WAVFile* song = (songBank != nullptr)
                                          ? songBank->songForIndex(voice->songIndex)
                                          : nullptr;
                if (song != nullptr) {
                    getLoopedSongSamples(*song,
                                         resources.inputBuffer.data[0],
                                         chunkSamples,
                                         static_cast<float>(runtimeConfig.sampleRate),
                                         voice->songCursorSamples);
                }
            } else {
                const ObjectAudioParams* toneParams = &voice->audioParams;
                const auto scheduledIt = toneParamsByObject.find(object.id);
                if (scheduledIt != toneParamsByObject.end()) {
                    toneParams = &scheduledIt->second;
                }
                getToneSamples(resources.inputBuffer.data[0],
                               chunkSamples,
                               static_cast<float>(audioTimeSec),
                               static_cast<float>(runtimeConfig.sampleRate),
                               *toneParams);
            }

            for (int i = 0; i < FRAME_SIZE; ++i) {
                resources.inputBuffer.data[0][i] *= object.fade;
            }

            const IPLfloat32 distanceAttenuation = iplDistanceAttenuationCalculate(
                resources.context, object.position, LISTENER_POSITION, &resources.distanceModel);
            IPLfloat32 airAbsorption[IPL_NUM_BANDS] = {1.0f, 1.0f, 1.0f};
            iplAirAbsorptionCalculate(resources.context,
                                      object.position,
                                      LISTENER_POSITION,
                                      &resources.airAbsorptionModel,
                                      airAbsorption);

            IPLDirectEffectParams directParams{};
            directParams.flags = static_cast<IPLDirectEffectFlags>(
                IPL_DIRECTEFFECTFLAGS_APPLYDISTANCEATTENUATION |
                IPL_DIRECTEFFECTFLAGS_APPLYAIRABSORPTION);
            directParams.transmissionType = IPL_TRANSMISSIONTYPE_FREQINDEPENDENT;
            directParams.distanceAttenuation = distanceAttenuation;
            directParams.directivity = 1.0f;
            directParams.occlusion = 0.0f;
            for (int band = 0; band < IPL_NUM_BANDS; ++band) {
                directParams.airAbsorption[band] = airAbsorption[band];
                directParams.transmission[band] = 1.0f;
            }

            iplDirectEffectApply(voice->directEffect,
                                 &directParams,
                                 &resources.inputBuffer,
                                 &resources.inputBuffer);

            zeroBuffer(resources.outputBuffer);
            IPLBinauralEffectParams binauralParams{};
            binauralParams.direction = direction;
            binauralParams.interpolation = IPL_HRTFINTERPOLATION_BILINEAR;
            binauralParams.spatialBlend = 1.0f;
            binauralParams.hrtf = resources.hrtf;

            iplBinauralEffectApply(voice->binauralEffect,
                                   &binauralParams,
                                   &resources.inputBuffer,
                                   &resources.outputBuffer);

            for (int i = 0; i < FRAME_SIZE; ++i) {
                resources.mixBuffer.data[0][i] += resources.outputBuffer.data[0][i];
                resources.mixBuffer.data[1][i] += resources.outputBuffer.data[1][i];
            }
        }

        interleaved.assign(static_cast<size_t>(chunkSamples) * 2, 0.0f);
        float peak = 0.0f;
        for (int i = 0; i < chunkSamples; ++i) {
            const float left = resources.mixBuffer.data[0][i] * runtimeConfig.masterGain;
            const float right = resources.mixBuffer.data[1][i] * runtimeConfig.masterGain;
            interleaved[2 * i] = left;
            interleaved[2 * i + 1] = right;
            peak = std::max(peak, std::max(std::abs(left), std::abs(right)));
        }
        if (peak > runtimeConfig.limiterThreshold && peak > 0.0f) {
            const float limiterGain = runtimeConfig.limiterThreshold / peak;
            for (size_t i = 0; i < interleaved.size(); ++i) {
                interleaved[i] *= limiterGain;
            }
        }

        const PaError err = Pa_WriteStream(stream, interleaved.data(), chunkSamples);
        if (err == paOutputUnderflowed) {
            ++outputUnderflowCount;
            if (gUnderflowWarningLimiter.shouldLog()) {
                std::cerr << fmt::format("Warning: PortAudio output underflow count={}\n",
                                         outputUnderflowCount);
            }
        } else if (err != paNoError) {
            std::cerr << fmt::format("Error: Pa_WriteStream failed: {}\n", Pa_GetErrorText(err));
            return false;
        }

        rendered += chunkSamples;
        audioTimeSec += static_cast<double>(chunkSamples) /
                        static_cast<double>(runtimeConfig.sampleRate);
    }

    return true;
}

void flushBinauralTails(PaStream* stream,
                        SteamAudioResources& resources,
                        std::unordered_map<int, SourceVoiceState>& voices) {
    if (stream == nullptr) {
        return;
    }

    std::vector<float> interleaved(static_cast<size_t>(FRAME_SIZE) * 2, 0.0f);

    for (auto& entry : voices) {
        SourceVoiceState& voice = entry.second;
        if (voice.binauralEffect == nullptr) {
            continue;
        }

        zeroBuffer(resources.outputBuffer);
        IPLAudioEffectState tailState =
            iplBinauralEffectGetTail(voice.binauralEffect, &resources.outputBuffer);

        int tailGuard = 0;
        while (tailState == IPL_AUDIOEFFECTSTATE_TAILREMAINING && tailGuard < 64) {
            for (int i = 0; i < FRAME_SIZE; ++i) {
                interleaved[2 * i] = resources.outputBuffer.data[0][i];
                interleaved[2 * i + 1] = resources.outputBuffer.data[1][i];
            }

            const PaError err = Pa_WriteStream(stream, interleaved.data(), FRAME_SIZE);
            if (err != paNoError && err != paOutputUnderflowed) {
                std::cerr << fmt::format(
                    "Warning: failed while draining tail for object {}: {}\n",
                    voice.id,
                    Pa_GetErrorText(err));
                break;
            }

            zeroBuffer(resources.outputBuffer);
            tailState = iplBinauralEffectGetTail(voice.binauralEffect, &resources.outputBuffer);
            ++tailGuard;
        }
    }
}

struct StreamClockState {
    bool anchored = false;
    double anchorSourceTimestampMs = 0.0;
    std::chrono::steady_clock::time_point anchorArrivalTime{};
    bool hasLastAcceptedTimestamp = false;
    double lastAcceptedTimestampMs = 0.0;
};

struct RuntimeDiagnostics {
    uint64_t timeoutNoFrameTicks = 0;
    uint64_t parseFailNoFrameTicks = 0;
    uint64_t staleDroppedFrames = 0;
    uint64_t staleAcceptedFrames = 0;
    double staleSumMs = 0.0;
    double staleMaxMs = 0.0;
    double maxInterpolationWindowUsedMs = 0.0;
};

double sanitizeSourceDeltaMs(double currentTimestampMs,
                             double previousTimestampMs,
                             bool hasPreviousTimestamp) {
    if (!hasPreviousTimestamp) {
        return DEFAULT_FRAME_INTERVAL_MS;
    }

    const double deltaMs = currentTimestampMs - previousTimestampMs;
    if (!std::isfinite(deltaMs) || deltaMs <= 0.0 || deltaMs > 1000.0) {
        return DEFAULT_FRAME_INTERVAL_MS;
    }
    return deltaMs;
}

double computeRenderDurationFromElapsedMs(double elapsedMs, int streamTimeoutMs) {
    const double maxRenderMs = std::max(1.0, static_cast<double>(streamTimeoutMs));
    if (!std::isfinite(elapsedMs)) {
        return maxRenderMs;
    }
    if (elapsedMs < 1.0) {
        return 1.0;
    }
    if (elapsedMs > maxRenderMs) {
        return maxRenderMs;
    }
    return elapsedMs;
}

int renderSamplesForDurationMs(double renderDurationMs, int sampleRate) {
    const int safeSampleRate = std::max(1, sampleRate);
    return std::max(
        1,
        static_cast<int>(
            std::llround((renderDurationMs / 1000.0) * static_cast<double>(safeSampleRate))));
}

uint64_t advanceTimelineUs(uint64_t timelineUs, double renderDurationMs) {
    if (!std::isfinite(renderDurationMs) || renderDurationMs <= 0.0) {
        return timelineUs;
    }
    const uint64_t tickAdvanceUs = static_cast<uint64_t>(std::llround(renderDurationMs * 1000.0));
    return timelineUs + tickAdvanceUs;
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

    RuntimeAudioConfig runtimeConfig;
    runtimeConfig.feedbackRateHz = options.feedbackRateHz;
    runtimeConfig.feedbackDutyCycle = options.feedbackDutyCycle;
    runtimeConfig.toneScheduleMode = options.toneScheduleMode;
    runtimeConfig.toneMinGapMs = options.toneMinGapMs;
    runtimeConfig.masterGain = options.masterGain;
    runtimeConfig.noFrameFadeMs = options.noFrameFadeMs;
    runtimeConfig.streamTimeoutMs = options.streamTimeoutMs;
    runtimeConfig.staleFrameDropMs = options.staleFrameDropMs;
    runtimeConfig.maxInterpWindowMs = options.maxInterpWindowMs;
    runtimeConfig.holdLastPositionMs = options.holdLastPositionMs;
    runtimeConfig.sourceMode = options.sourceMode;
    runtimeConfig.songAPath = options.songAPath;
    runtimeConfig.songBPath = options.songBPath;

    jsa::core::ResourceLocator resourceLocator;
    if (!options.assetsRoot.empty()) {
        resourceLocator.setAssetsRoot(options.assetsRoot);
    }

    std::string resolveErr;
    std::string customHrtfPath = DEFAULT_HRTF_SOFA;
    if (!options.useDefaultHRTF) {
        const auto resolvedHrtf = resourceLocator.resolveAsset(customHrtfPath, resolveErr);
        if (!resolvedHrtf.has_value()) {
            std::cerr << fmt::format("Error: {}\n", resolveErr);
            return 1;
        }
        customHrtfPath = *resolvedHrtf;
    }

    SongBank songBank;
    if (runtimeConfig.sourceMode == AudioSourceMode::Songs) {
        const auto resolvedSongA = resourceLocator.resolveAsset(runtimeConfig.songAPath, resolveErr);
        if (!resolvedSongA.has_value()) {
            std::cerr << fmt::format("Error: {}\n", resolveErr);
            return 1;
        }
        const auto resolvedSongB = resourceLocator.resolveAsset(runtimeConfig.songBPath, resolveErr);
        if (!resolvedSongB.has_value()) {
            std::cerr << fmt::format("Error: {}\n", resolveErr);
            return 1;
        }
        runtimeConfig.songAPath = *resolvedSongA;
        runtimeConfig.songBPath = *resolvedSongB;

        if (!readWAV(runtimeConfig.songAPath, songBank.songA)) {
            return 1;
        }
        if (!readWAV(runtimeConfig.songBPath, songBank.songB)) {
            return 1;
        }
    }

    PaStream* stream = nullptr;
    if (!initializeOutputStream(options.deviceIndex,
                                options.sampleRate,
                                stream,
                                runtimeConfig.sampleRate)) {
        return 1;
    }

    SteamAudioResources steamAudio;
    if (!initializeSteamAudioResources(options.useDefaultHRTF,
                                       runtimeConfig.sampleRate,
                                       customHrtfPath,
                                       steamAudio)) {
        shutdownOutputStream(stream);
        return 1;
    }

    if (runtimeConfig.sourceMode == AudioSourceMode::Songs) {
        std::cout << fmt::format(
            "Song source mode: A={}, B={}, engine sample rate: {} Hz, master gain: {:.2f}, no-frame fade: {:.1f} ms, timeout: {} ms, stale drop: {:.1f} ms, interp max: {:.1f} ms, hold-last: {:.1f} ms\n",
            runtimeConfig.songAPath,
            runtimeConfig.songBPath,
            runtimeConfig.sampleRate,
            runtimeConfig.masterGain,
            runtimeConfig.noFrameFadeMs,
            runtimeConfig.streamTimeoutMs,
            runtimeConfig.staleFrameDropMs,
            runtimeConfig.maxInterpWindowMs,
            runtimeConfig.holdLastPositionMs);
    } else {
        const char* toneScheduleLabel =
            (runtimeConfig.toneScheduleMode == ToneScheduleMode::RoundRobin)
                ? "round-robin"
                : "hash";
        std::cout << fmt::format(
            "Synth source mode: base {:.3f} Hz, engine sample rate: {} Hz, feedback rate: {:.2f} Hz, duty: {:.2f}, tone schedule: {}, tone min gap: {:.1f} ms, master gain: {:.2f}, no-frame fade: {:.1f} ms, timeout: {} ms, stale drop: {:.1f} ms, interp max: {:.1f} ms, hold-last: {:.1f} ms\n",
            DEFAULT_TONE_BASE_FREQ_HZ,
            runtimeConfig.sampleRate,
            runtimeConfig.feedbackRateHz,
            runtimeConfig.feedbackDutyCycle,
            toneScheduleLabel,
            runtimeConfig.toneMinGapMs,
            runtimeConfig.masterGain,
            runtimeConfig.noFrameFadeMs,
            runtimeConfig.streamTimeoutMs,
            runtimeConfig.staleFrameDropMs,
            runtimeConfig.maxInterpWindowMs,
            runtimeConfig.holdLastPositionMs);
    }

    const SongBank* activeSongBank =
        (runtimeConfig.sourceMode == AudioSourceMode::Songs) ? &songBank : nullptr;

    prepareIpcEndpoint(options.ipcEndpoint);

    void* zmqContext = zmq_ctx_new();
    if (zmqContext == nullptr) {
        std::cerr << "Error: failed to create ZeroMQ context\n";
        shutdownOutputStream(stream);
        return 1;
    }

    void* socket = zmq_socket(zmqContext, ZMQ_REP);
    if (socket == nullptr) {
        std::cerr << "Error: failed to create ZeroMQ REP socket\n";
        zmq_ctx_term(zmqContext);
        shutdownOutputStream(stream);
        return 1;
    }

    const int lingerMs = 0;
    zmq_setsockopt(socket, ZMQ_LINGER, &lingerMs, sizeof(lingerMs));
    const int receiveTimeoutMs = runtimeConfig.streamTimeoutMs;
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &receiveTimeoutMs, sizeof(receiveTimeoutMs));

    if (zmq_bind(socket, options.ipcEndpoint.c_str()) != 0) {
        std::cerr << fmt::format("Error: failed to bind socket {}: {}\n",
                                 options.ipcEndpoint,
                                 zmq_strerror(zmq_errno()));
        zmq_close(socket);
        zmq_ctx_term(zmqContext);
        shutdownOutputStream(stream);
        return 1;
    }

    std::cout << fmt::format("Listening on {}\n", options.ipcEndpoint);
    std::cout << "Press Ctrl+C to stop.\n";

    ObjectTracker3D tracker(runtimeConfig.sampleRate,
                            runtimeConfig.noFrameFadeMs,
                            runtimeConfig.holdLastPositionMs);
    std::unordered_map<int, SourceVoiceState> voices;

    double audioTimeSec = 0.0;
    uint64_t receivedFrames = 0;
    uint64_t outputUnderflowCount = 0;
    uint64_t renderTimelineUs = 0;
    bool hasPreviousSourceTimestamp = false;
    double previousSourceTimestampMs = 0.0;
    StreamClockState streamClock;
    RuntimeDiagnostics diagnostics;
    auto lastRenderTick = std::chrono::steady_clock::now();

    while (gRunning.load()) {
        zmq_msg_t message;
        zmq_msg_init(&message);

        const int recvResult = zmq_msg_recv(&message, socket, 0);
        const auto eventTime = std::chrono::steady_clock::now();
        const double elapsedMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                     eventTime - lastRenderTick)
                                     .count();
        lastRenderTick = eventTime;
        const double renderDurationMs =
            computeRenderDurationFromElapsedMs(elapsedMs, runtimeConfig.streamTimeoutMs);
        const int renderSamples =
            renderSamplesForDurationMs(renderDurationMs, runtimeConfig.sampleRate);
        renderTimelineUs = advanceTimelineUs(renderTimelineUs, renderDurationMs);

        double interpolationHintMs = renderDurationMs;
        bool frameAccepted = false;
        int latestFrameNumber = -1;
        size_t latestObjectCount = 0;

        if (recvResult == -1) {
            const int errnum = zmq_errno();
            zmq_msg_close(&message);
            if (errnum == EAGAIN || errnum == EINTR) {
                ++diagnostics.timeoutNoFrameTicks;
                tracker.updateWithoutFrame(renderTimelineUs, static_cast<float>(renderSamples));
            } else {
                std::cerr << fmt::format("Error: socket receive failed: {}\n", zmq_strerror(errnum));
                break;
            }
        } else {
            const auto* payload = static_cast<const uint8_t*>(zmq_msg_data(&message));
            const size_t payloadLen = zmq_msg_size(&message);

            SocketFrame3D frameData;
            std::string parseError;
            const bool parsed = parseSocketObjectRep3D(payload, payloadLen, frameData, parseError);
            zmq_msg_close(&message);

            if (!parsed) {
                if (zmq_send(socket, "1", 1, 0) == -1) {
                    std::cerr << "Error: failed to send parse failure ack\n";
                    break;
                }
                ++diagnostics.parseFailNoFrameTicks;
                if (gParseWarningLimiter.shouldLog()) {
                    std::cerr << fmt::format(
                        "Parse failure: {} (payload {} bytes)\n", parseError, payloadLen);
                }
                tracker.updateWithoutFrame(renderTimelineUs, static_cast<float>(renderSamples));
            } else {
                if (zmq_send(socket, "0", 1, 0) == -1) {
                    std::cerr << "Error: failed to send success ack\n";
                    break;
                }

                const double backwardResetThresholdMs =
                    std::max(100.0, static_cast<double>(runtimeConfig.staleFrameDropMs));
                const bool hasBackwardJump =
                    streamClock.hasLastAcceptedTimestamp &&
                    frameData.timestamp_ms + backwardResetThresholdMs <
                        streamClock.lastAcceptedTimestampMs;
                if (!streamClock.anchored || hasBackwardJump) {
                    if (hasBackwardJump && gStaleDropWarningLimiter.shouldLog()) {
                        std::cerr << fmt::format(
                            "Info: source timestamp jump detected (prev {:.1f} ms, now {:.1f} ms). Resetting stream clock anchor.\n",
                            streamClock.lastAcceptedTimestampMs,
                            frameData.timestamp_ms);
                    }
                    streamClock.anchored = true;
                    streamClock.anchorSourceTimestampMs = frameData.timestamp_ms;
                    streamClock.anchorArrivalTime = eventTime;
                }

                double stalenessMs = 0.0;
                if (streamClock.anchored) {
                    const double elapsedSinceAnchorMs =
                        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                            eventTime - streamClock.anchorArrivalTime)
                            .count();
                    const double expectedSourceNowMs =
                        streamClock.anchorSourceTimestampMs + elapsedSinceAnchorMs;
                    stalenessMs = expectedSourceNowMs - frameData.timestamp_ms;
                    if (!std::isfinite(stalenessMs)) {
                        stalenessMs = 0.0;
                    }
                }

                if (stalenessMs > static_cast<double>(runtimeConfig.staleFrameDropMs)) {
                    ++diagnostics.staleDroppedFrames;
                    if (gStaleDropWarningLimiter.shouldLog()) {
                        std::cerr << fmt::format(
                            "Warning: dropping stale frame {} (staleness {:.1f} ms > {:.1f} ms)\n",
                            frameData.frame_number,
                            stalenessMs,
                            runtimeConfig.staleFrameDropMs);
                    }
                    tracker.updateWithoutFrame(renderTimelineUs, static_cast<float>(renderSamples));
                } else {
                    interpolationHintMs = sanitizeSourceDeltaMs(
                        frameData.timestamp_ms,
                        previousSourceTimestampMs,
                        hasPreviousSourceTimestamp);
                    previousSourceTimestampMs = frameData.timestamp_ms;
                    hasPreviousSourceTimestamp = true;

                    const double stalenessForStatsMs = std::max(0.0, stalenessMs);
                    ++diagnostics.staleAcceptedFrames;
                    diagnostics.staleSumMs += stalenessForStatsMs;
                    diagnostics.staleMaxMs = std::max(diagnostics.staleMaxMs, stalenessForStatsMs);

                    streamClock.hasLastAcceptedTimestamp = true;
                    streamClock.lastAcceptedTimestampMs = frameData.timestamp_ms;

                    tracker.updateFromFrame(frameData, renderTimelineUs, static_cast<float>(renderSamples));
                    latestFrameNumber = frameData.frame_number;
                    latestObjectCount = frameData.objects.size();
                    frameAccepted = true;
                    ++receivedFrames;
                }
            }
        }

        const std::vector<int> releasable = tracker.collectReleasableObjects(renderTimelineUs);
        for (const int objectId : releasable) {
            releaseVoiceForObject(objectId, voices);
        }

        if (!renderAndPlaySamples(tracker,
                                  renderSamples,
                                  interpolationHintMs,
                                  audioTimeSec,
                                  runtimeConfig,
                                  activeSongBank,
                                  steamAudio,
                                  stream,
                                  voices,
                                  diagnostics.maxInterpolationWindowUsedMs,
                                  outputUnderflowCount)) {
            break;
        }

        if (frameAccepted && receivedFrames % 30 == 0) {
            std::cout << fmt::format(
                "Processed {} accepted frames (latest frame {}, objects: {}, voices: {}, underflows: {}, stale dropped: {})\n",
                receivedFrames,
                latestFrameNumber,
                latestObjectCount,
                voices.size(),
                outputUnderflowCount,
                diagnostics.staleDroppedFrames);
        }

        if (gTimingStatsLimiter.shouldLog()) {
            const double averageStalenessMs =
                (diagnostics.staleAcceptedFrames > 0)
                    ? (diagnostics.staleSumMs /
                       static_cast<double>(diagnostics.staleAcceptedFrames))
                    : 0.0;
            std::cout << fmt::format(
                "Timing stats: stale_drop={}, timeout_ticks={}, parse_fail_ticks={}, stale_avg={:.1f} ms, stale_max={:.1f} ms, interp_window_max={:.1f} ms\n",
                diagnostics.staleDroppedFrames,
                diagnostics.timeoutNoFrameTicks,
                diagnostics.parseFailNoFrameTicks,
                averageStalenessMs,
                diagnostics.staleMaxMs,
                diagnostics.maxInterpolationWindowUsedMs);
        }
    }

    std::cout << "Shutting down...\n";
    std::cout << fmt::format("Total PortAudio output underflows: {}\n", outputUnderflowCount);
    flushBinauralTails(stream, steamAudio, voices);

    for (auto& entry : voices) {
        releaseVoice(entry.second);
    }
    voices.clear();

    zmq_close(socket);
    zmq_ctx_term(zmqContext);
    shutdownOutputStream(stream);
    steamAudio.release();
    return 0;
}
