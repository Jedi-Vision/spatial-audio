#include <phonon.h>
#include <portaudio.h>
#include <zmq.h>

#include <fmt/format.h>

#include <jsa/protocol/frame_parser.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr int MAX_SIMULTANEOUS_OBJECTS = 4;
constexpr int SAMPLE_RATE = 48000;
constexpr int FRAME_SIZE = 1024;
constexpr float DEFAULT_FRAME_INTERVAL_MS = 33.333333f;
constexpr float CAMERA_FOV_HORIZONTAL_DEG = 60.0f;
constexpr float CAMERA_FOV_VERTICAL_DEG = 45.0f;
constexpr float FADE_IN_TIME_MS = 100.0f;
constexpr float FADE_OUT_TIME_MS = 100.0f;

constexpr float BEAT_CYCLE_MS = 2000.0f;
constexpr float BEAT_DURATION_MS = 400.0f;
constexpr float SOURCE_AUDIO_LIMIT_MS = 300.0f;
constexpr float BEAT_OFFSET[MAX_SIMULTANEOUS_OBJECTS] = {0.0f, 0.25f, 0.5f, 0.75f};
constexpr float PITCH_SEMITONES[MAX_SIMULTANEOUS_OBJECTS] = {0.0f, 4.0f, 7.0f, 11.0f};

constexpr const char* DEFAULT_IPC_ENDPOINT = "ipc:///tmp/jv/audio/0.sock";
constexpr const char* DEFAULT_AUDIO_FILE = "beep_1.wav";
constexpr const char* DEFAULT_HRTF_SOFA = "D2_HRIR_SOFA/D2_44K_16bit_256tap_FIR_SOFA.sofa";

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

WarningLimiter gXWarningLimiter;
WarningLimiter gYWarningLimiter;

struct CLIOptions {
    std::string ipcEndpoint = DEFAULT_IPC_ENDPOINT;
    std::string audioPath = DEFAULT_AUDIO_FILE;
    bool useDefaultHRTF = true;
    int deviceIndex = -1;
    bool showHelp = false;
};

struct WAVHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunkSize = 0;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmtSize = 16;
    uint16_t audioFormat = 1;
    uint16_t numChannels = 1;
    uint32_t sampleRate = 0;
    uint32_t byteRate = 0;
    uint16_t blockAlign = 0;
    uint16_t bitsPerSample = 0;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize = 0;
};

struct WAVFile {
    uint32_t sampleRate = 0;
    uint16_t numChannels = 0;
    uint16_t bitsPerSample = 0;
    std::vector<float> samples;
};

struct ObjectAudioParams {
    float pitchRatio = 1.0f;
    float beatOffset = 0.0f;
    float beatCycleSec = 0.6f;
    float beatDurationSec = 0.15f;

    static ObjectAudioParams forSlot(int slot) {
        ObjectAudioParams params;
        const float semitones = (slot >= 0 && slot < MAX_SIMULTANEOUS_OBJECTS)
                                    ? PITCH_SEMITONES[slot]
                                    : 0.0f;
        params.pitchRatio = std::pow(2.0f, semitones / 12.0f);
        params.beatOffset = (slot >= 0 && slot < MAX_SIMULTANEOUS_OBJECTS)
                                ? BEAT_OFFSET[slot]
                                : 0.0f;
        params.beatCycleSec = BEAT_CYCLE_MS / 1000.0f;
        params.beatDurationSec = BEAT_DURATION_MS / 1000.0f;
        return params;
    }
};

struct TrackedObject {
    int id = -1;
    IPLVector3 currentPosition = {0.0f, 0.0f, -1.0f};
    IPLVector3 previousPosition = {0.0f, 0.0f, -1.0f};
    float currentDistance = 1.0f;
    float previousDistance = 1.0f;
    float fadeVolume = 0.0f;
    bool active = false;
    uint64_t lastSeenTimestampUs = 0;
};

struct SteamAudioResources {
    IPLContext context = nullptr;
    IPLHRTF hrtf = nullptr;
    IPLAudioSettings audioSettings{};
    IPLDistanceAttenuationModel distanceModel{};
    IPLAirAbsorptionModel airAbsorptionModel{};
    std::vector<IPLBinauralEffect> binauralEffects;
    std::vector<IPLDirectEffect> directEffects;
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
    std::cout << "  --audio <wav>               Mono source WAV path (default: beep_1.wav)\n";
    std::cout << "  --hrtf <default|custom>     HRTF type (default: default)\n";
    std::cout << "  --device-index <index>      PortAudio output device index (default: -1)\n";
    std::cout << "  --help, -h                  Show this help message\n";
}

CLIOptions parseCommandLine(int argc, char* argv[]) {
    CLIOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if ((arg == "--help") || (arg == "-h")) {
            options.showHelp = true;
            continue;
        }

        if (arg == "--ipc" && i + 1 < argc) {
            options.ipcEndpoint = argv[++i];
            continue;
        }
        if (arg == "--audio" && i + 1 < argc) {
            options.audioPath = argv[++i];
            continue;
        }
        if (arg == "--hrtf" && i + 1 < argc) {
            const std::string_view hrtfType(argv[++i]);
            options.useDefaultHRTF = (hrtfType == "default");
            continue;
        }
        if (arg == "--device-index" && i + 1 < argc) {
            options.deviceIndex = std::atoi(argv[++i]);
            continue;
        }

        std::cerr << fmt::format("Unknown or incomplete option: {}\n", arg);
        options.showHelp = true;
        break;
    }

    return options;
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

    for (auto& effect : directEffects) {
        if (effect != nullptr) {
            iplDirectEffectRelease(&effect);
        }
    }
    directEffects.clear();

    for (auto& effect : binauralEffects) {
        if (effect != nullptr) {
            iplBinauralEffectRelease(&effect);
        }
    }
    binauralEffects.clear();

    if (hrtf != nullptr) {
        iplHRTFRelease(&hrtf);
        hrtf = nullptr;
    }

    if (context != nullptr) {
        iplContextRelease(&context);
        context = nullptr;
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

bool readWAV(const std::string& filename, WAVFile& wav) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << fmt::format("Error: could not open WAV file: {}\n", filename);
        return false;
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    if (std::memcmp(header.riff, "RIFF", 4) != 0 || std::memcmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "Error: not a valid WAV file\n";
        return false;
    }

    if (std::memcmp(header.fmt, "fmt ", 4) != 0) {
        file.seekg(12);
        char chunkID[4] = {};
        uint32_t chunkSize = 0;
        bool foundFmt = false;
        while (file.read(chunkID, 4) && file.read(reinterpret_cast<char*>(&chunkSize), 4)) {
            if (std::memcmp(chunkID, "fmt ", 4) == 0) {
                file.read(reinterpret_cast<char*>(&header.audioFormat), 2);
                file.read(reinterpret_cast<char*>(&header.numChannels), 2);
                file.read(reinterpret_cast<char*>(&header.sampleRate), 4);
                file.read(reinterpret_cast<char*>(&header.byteRate), 4);
                file.read(reinterpret_cast<char*>(&header.blockAlign), 2);
                file.read(reinterpret_cast<char*>(&header.bitsPerSample), 2);
                if (chunkSize > 16) {
                    file.seekg(static_cast<std::streamoff>(chunkSize - 16), std::ios::cur);
                }
                foundFmt = true;
                break;
            }
            file.seekg(static_cast<std::streamoff>(chunkSize), std::ios::cur);
        }

        if (!foundFmt) {
            std::cerr << "Error: could not find fmt chunk\n";
            return false;
        }

        while (file.read(chunkID, 4) && file.read(reinterpret_cast<char*>(&header.dataSize), 4)) {
            if (std::memcmp(chunkID, "data", 4) == 0) {
                break;
            }
            file.seekg(static_cast<std::streamoff>(header.dataSize), std::ios::cur);
        }

        if (std::memcmp(chunkID, "data", 4) != 0) {
            std::cerr << "Error: could not find data chunk\n";
            return false;
        }
    }

    if (header.audioFormat != 1) {
        std::cerr << "Error: only PCM WAV is supported\n";
        return false;
    }

    wav.sampleRate = header.sampleRate;
    wav.numChannels = header.numChannels;
    wav.bitsPerSample = header.bitsPerSample;

    const int bytesPerSample = header.bitsPerSample / 8;
    if (bytesPerSample <= 0) {
        std::cerr << fmt::format("Error: unsupported bits per sample: {}\n", header.bitsPerSample);
        return false;
    }

    const size_t numSamples = header.dataSize / (bytesPerSample * header.numChannels);
    std::vector<uint8_t> rawData(header.dataSize);
    file.read(reinterpret_cast<char*>(rawData.data()),
              static_cast<std::streamsize>(rawData.size()));

    if (static_cast<size_t>(file.gcount()) != rawData.size()) {
        std::cerr << "Error: failed to read full data chunk\n";
        return false;
    }

    auto sampleToFloat = [&](const uint8_t* ptr) -> float {
        switch (header.bitsPerSample) {
            case 8: {
                const int32_t value = static_cast<int32_t>(*ptr) - 128;
                return static_cast<float>(value) / 128.0f;
            }
            case 16: {
                int32_t value = ptr[0] | (ptr[1] << 8);
                if (value & 0x8000) {
                    value |= ~0xFFFF;
                }
                return static_cast<float>(value) / 32768.0f;
            }
            case 24: {
                int32_t value = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16);
                if (value & 0x800000) {
                    value |= ~0xFFFFFF;
                }
                return static_cast<float>(value) / 8388608.0f;
            }
            case 32: {
                const int32_t value =
                    ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
                return static_cast<float>(value) / 2147483648.0f;
            }
            default:
                return 0.0f;
        }
    };

    wav.samples.resize(numSamples);
    for (size_t i = 0; i < numSamples; ++i) {
        float mixed = 0.0f;
        for (uint16_t channel = 0; channel < header.numChannels; ++channel) {
            const size_t offset = (i * header.numChannels + channel) * bytesPerSample;
            mixed += sampleToFloat(&rawData[offset]);
        }
        wav.samples[i] = mixed / static_cast<float>(header.numChannels);
    }

    std::cout << fmt::format("Loaded WAV: {} ({} Hz, {} samples)\n",
                             filename,
                             wav.sampleRate,
                             wav.samples.size());
    return true;
}

float clampNormalized(double value, const char* axisName) {
    if (!std::isfinite(value)) {
        if ((axisName[0] == 'x' && gXWarningLimiter.shouldLog()) ||
            (axisName[0] == 'y' && gYWarningLimiter.shouldLog())) {
            std::cerr << fmt::format("Warning: non-finite {} coordinate, clamping to 0.5\n",
                                     axisName);
        }
        return 0.5f;
    }

    if (value < 0.0 || value > 1.0) {
        if ((axisName[0] == 'x' && gXWarningLimiter.shouldLog()) ||
            (axisName[0] == 'y' && gYWarningLimiter.shouldLog())) {
            std::cerr << fmt::format(
                "Warning: {} coordinate out of range [{:.4f}], clamping to [0,1]\n",
                axisName,
                value);
        }
    }

    float clamped = static_cast<float>(value);
    if (clamped < 0.0f) {
        clamped = 0.0f;
    } else if (clamped > 1.0f) {
        clamped = 1.0f;
    }
    return clamped;
}

IPLVector3 convertToWorldSpace(float x2d, float y2d, float depth) {
    constexpr float kPi = 3.14159265358979323846f;
    const float normalizedX = x2d - 0.5f;
    const float normalizedY = 0.5f - y2d;
    const float fovHRad = CAMERA_FOV_HORIZONTAL_DEG * kPi / 180.0f;
    const float fovVRad = CAMERA_FOV_VERTICAL_DEG * kPi / 180.0f;

    const float horizontalAngle = normalizedX * fovHRad;
    const float verticalAngle = normalizedY * fovVRad;

    IPLVector3 position{};
    position.x = depth * std::tan(horizontalAngle);
    position.y = depth * std::tan(verticalAngle);
    position.z = -depth;
    return position;
}

class ObjectTracker {
public:
    ObjectTracker()
        : fadeInSamples((FADE_IN_TIME_MS / 1000.0f) * SAMPLE_RATE),
          fadeOutSamples((FADE_OUT_TIME_MS / 1000.0f) * SAMPLE_RATE) {}

    void updateFromFrame(const SocketFrameData& frame,
                         uint64_t currentTimeUs,
                         float updateSamples) {
        for (auto& pair : trackedObjects) {
            pair.second.active = false;
        }

        const float fadeInStep = std::max(0.0f, updateSamples / fadeInSamples);
        const float fadeOutStep = std::max(0.0f, updateSamples / fadeOutSamples);

        for (const auto& obj : frame.objects) {
            const float xNorm = clampNormalized(obj.x_2d, "x");
            const float yNorm = clampNormalized(obj.y_2d, "y");

            float depth = static_cast<float>(obj.depth);
            if (!std::isfinite(depth) || depth <= 0.0f) {
                depth = 0.1f;
            }

            const IPLVector3 position = convertToWorldSpace(xNorm, yNorm, depth);

            auto it = trackedObjects.find(obj.id);
            if (it != trackedObjects.end()) {
                TrackedObject& tracked = it->second;
                tracked.previousPosition = tracked.currentPosition;
                tracked.previousDistance = tracked.currentDistance;
                tracked.currentPosition = position;
                tracked.currentDistance = depth;
                tracked.active = true;
                tracked.lastSeenTimestampUs = currentTimeUs;
                tracked.fadeVolume = std::min(1.0f, tracked.fadeVolume + fadeInStep);
            } else {
                TrackedObject tracked;
                tracked.id = obj.id;
                tracked.currentPosition = position;
                tracked.previousPosition = position;
                tracked.currentDistance = depth;
                tracked.previousDistance = depth;
                tracked.active = true;
                tracked.fadeVolume = std::min(1.0f, fadeInStep);
                tracked.lastSeenTimestampUs = currentTimeUs;
                trackedObjects[obj.id] = tracked;
            }
        }

        for (auto& pair : trackedObjects) {
            TrackedObject& tracked = pair.second;
            if (!tracked.active) {
                tracked.fadeVolume = std::max(0.0f, tracked.fadeVolume - fadeOutStep);
            }
        }

        for (auto it = trackedObjects.begin(); it != trackedObjects.end();) {
            const TrackedObject& tracked = it->second;
            if (!tracked.active && tracked.fadeVolume <= 0.0f &&
                currentTimeUs > tracked.lastSeenTimestampUs &&
                (currentTimeUs - tracked.lastSeenTimestampUs) > 2000000ULL) {
                it = trackedObjects.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::vector<std::pair<int, IPLVector3>> getInterpolatedPositions(float interpolationFactor) const {
        std::vector<std::pair<int, IPLVector3>> positions;

        for (const auto& pair : trackedObjects) {
            const TrackedObject& tracked = pair.second;
            if (tracked.fadeVolume <= 0.0f) {
                continue;
            }

            IPLVector3 interpolated{};
            interpolated.x = tracked.previousPosition.x +
                             (tracked.currentPosition.x - tracked.previousPosition.x) *
                                 interpolationFactor;
            interpolated.y = tracked.previousPosition.y +
                             (tracked.currentPosition.y - tracked.previousPosition.y) *
                                 interpolationFactor;
            interpolated.z = tracked.previousPosition.z +
                             (tracked.currentPosition.z - tracked.previousPosition.z) *
                                 interpolationFactor;
            positions.push_back({tracked.id, interpolated});
        }

        if (positions.size() > MAX_SIMULTANEOUS_OBJECTS) {
            std::sort(positions.begin(),
                      positions.end(),
                      [this](const auto& a, const auto& b) {
                          return getFadeVolume(a.first) > getFadeVolume(b.first);
                      });
            positions.resize(MAX_SIMULTANEOUS_OBJECTS);
        }

        return positions;
    }

    float getFadeVolume(int objectId) const {
        const auto it = trackedObjects.find(objectId);
        if (it == trackedObjects.end()) {
            return 0.0f;
        }
        return it->second.fadeVolume;
    }

private:
    std::unordered_map<int, TrackedObject> trackedObjects;
    float fadeInSamples = 0.0f;
    float fadeOutSamples = 0.0f;
};

void getAudioSamples(const WAVFile& wav,
                     float* output,
                     int numSamples,
                     float startTimeSec,
                     float targetSampleRate,
                     float pitchRatio,
                     float beatOffset,
                     float beatCycleSec,
                     float beatDurationSec) {
    if (wav.samples.empty()) {
        std::fill(output, output + numSamples, 0.0f);
        return;
    }

    const float sourceSampleRate = static_cast<float>(wav.sampleRate);
    const size_t sourceSize = wav.samples.size();
    const float slotStart = beatOffset * beatCycleSec;
    const float slotEnd = slotStart + beatDurationSec;
    const float sourceLimitSamples = (SOURCE_AUDIO_LIMIT_MS / 1000.0f) * sourceSampleRate;
    const float effectiveSourceSize =
        std::min(static_cast<float>(sourceSize), sourceLimitSamples);

    for (int i = 0; i < numSamples; ++i) {
        const float targetTime = startTimeSec + (static_cast<float>(i) / targetSampleRate);
        const float cyclePosition = std::fmod(targetTime, beatCycleSec);

        bool inSlot = false;
        if (slotEnd <= beatCycleSec) {
            inSlot = (cyclePosition >= slotStart && cyclePosition < slotEnd);
        } else {
            inSlot = (cyclePosition >= slotStart ||
                      cyclePosition < std::fmod(slotEnd, beatCycleSec));
        }

        if (!inSlot) {
            output[i] = 0.0f;
            continue;
        }

        float timeInSlot = cyclePosition - slotStart;
        if (timeInSlot < 0.0f) {
            timeInSlot += beatCycleSec;
        }

        const float sourceSampleIndex = timeInSlot * sourceSampleRate * pitchRatio;
        if (sourceSampleIndex >= effectiveSourceSize) {
            output[i] = 0.0f;
            continue;
        }

        float envelope = 1.0f;
        const float fadeTime = 0.01f;
        const float sourceTime = sourceSampleIndex / sourceSampleRate;
        const float sourceDuration = effectiveSourceSize / sourceSampleRate;
        if (sourceTime < fadeTime) {
            envelope = sourceTime / fadeTime;
        } else if (sourceTime > sourceDuration - fadeTime) {
            envelope = (sourceDuration - sourceTime) / fadeTime;
        }

        const size_t index = static_cast<size_t>(sourceSampleIndex);
        const size_t nextIndex =
            std::min(index + 1, static_cast<size_t>(effectiveSourceSize) - 1);
        const float fraction = sourceSampleIndex - std::floor(sourceSampleIndex);
        const float sample = wav.samples[index] +
                             (wav.samples[nextIndex] - wav.samples[index]) * fraction;
        output[i] = sample * envelope;
    }
}

bool createHRTF(bool useDefault, SteamAudioResources& resources) {
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
        const std::string sofaPath = DEFAULT_HRTF_SOFA;
        hrtfSettings.sofaFileName = sofaPath.c_str();
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

bool initializeSteamAudioResources(bool useDefaultHRTF, SteamAudioResources& resources) {
    resources.release();

    IPLContextSettings contextSettings{};
    contextSettings.version = STEAMAUDIO_VERSION;
    contextSettings.simdLevel = IPL_SIMDLEVEL_NEON;

    IPLerror error = iplContextCreate(&contextSettings, &resources.context);
    if (error != IPL_STATUS_SUCCESS) {
        std::cerr << "Error: failed to create Steam Audio context\n";
        return false;
    }

    resources.audioSettings.samplingRate = SAMPLE_RATE;
    resources.audioSettings.frameSize = FRAME_SIZE;

    if (!createHRTF(useDefaultHRTF, resources)) {
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

    resources.binauralEffects.assign(MAX_SIMULTANEOUS_OBJECTS, nullptr);
    resources.directEffects.assign(MAX_SIMULTANEOUS_OBJECTS, nullptr);

    IPLBinauralEffectSettings binauralSettings{};
    binauralSettings.hrtf = resources.hrtf;
    IPLDirectEffectSettings directSettings{};
    directSettings.numChannels = 1;

    for (int i = 0; i < MAX_SIMULTANEOUS_OBJECTS; ++i) {
        error = iplBinauralEffectCreate(resources.context,
                                        &resources.audioSettings,
                                        &binauralSettings,
                                        &resources.binauralEffects[i]);
        if (error != IPL_STATUS_SUCCESS) {
            std::cerr << fmt::format("Error: failed to create binaural effect {}\n", i);
            resources.release();
            return false;
        }

        error = iplDirectEffectCreate(resources.context,
                                      &resources.audioSettings,
                                      &directSettings,
                                      &resources.directEffects[i]);
        if (error != IPL_STATUS_SUCCESS) {
            std::cerr << fmt::format("Error: failed to create direct effect {}\n", i);
            resources.release();
            return false;
        }
    }

    return true;
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

bool initializeOutputStream(int requestedDeviceIndex, PaStream*& stream) {
    stream = nullptr;

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

    PaStreamParameters outputParameters{};
    outputParameters.device = outputDevice;
    outputParameters.channelCount = 2;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency = deviceInfo->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(&stream,
                        nullptr,
                        &outputParameters,
                        static_cast<double>(SAMPLE_RATE),
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

int acquireEffectSlot(int objectId,
                      std::unordered_map<int, int>& objectToSlot,
                      std::array<int, MAX_SIMULTANEOUS_OBJECTS>& slotToObject,
                      const SteamAudioResources& resources) {
    const auto existing = objectToSlot.find(objectId);
    if (existing != objectToSlot.end()) {
        return existing->second;
    }

    for (int slot = 0; slot < MAX_SIMULTANEOUS_OBJECTS; ++slot) {
        if (slotToObject[slot] == -1) {
            slotToObject[slot] = objectId;
            objectToSlot[objectId] = slot;
            iplBinauralEffectReset(resources.binauralEffects[slot]);
            iplDirectEffectReset(resources.directEffects[slot]);
            return slot;
        }
    }

    return -1;
}

void releaseFadedSlots(const ObjectTracker& tracker,
                       std::unordered_map<int, int>& objectToSlot,
                       std::array<int, MAX_SIMULTANEOUS_OBJECTS>& slotToObject) {
    for (auto it = objectToSlot.begin(); it != objectToSlot.end();) {
        if (tracker.getFadeVolume(it->first) <= 0.0f) {
            slotToObject[it->second] = -1;
            it = objectToSlot.erase(it);
        } else {
            ++it;
        }
    }
}

bool renderAndPlaySamples(const ObjectTracker& tracker,
                          int samplesToRender,
                          double& audioTimeSec,
                          const WAVFile& sourceAudio,
                          SteamAudioResources& resources,
                          PaStream* stream,
                          std::unordered_map<int, int>& objectToSlot,
                          std::array<int, MAX_SIMULTANEOUS_OBJECTS>& slotToObject) {
    if (samplesToRender <= 0) {
        return true;
    }

    int rendered = 0;
    std::vector<float> interleaved;
    interleaved.reserve(static_cast<size_t>(FRAME_SIZE) * 2);

    while (rendered < samplesToRender && gRunning.load()) {
        const int chunkSamples = std::min(FRAME_SIZE, samplesToRender - rendered);
        const float interpolationFactor =
            static_cast<float>(rendered + chunkSamples) / static_cast<float>(samplesToRender);

        releaseFadedSlots(tracker, objectToSlot, slotToObject);
        const auto activeObjects = tracker.getInterpolatedPositions(interpolationFactor);

        zeroBuffer(resources.mixBuffer);

        for (const auto& entry : activeObjects) {
            const int objectId = entry.first;
            const IPLVector3& position = entry.second;
            const int effectSlot =
                acquireEffectSlot(objectId, objectToSlot, slotToObject, resources);
            if (effectSlot < 0) {
                continue;
            }

            const float distance = std::max(
                0.001f,
                std::sqrt(position.x * position.x + position.y * position.y + position.z * position.z));
            IPLVector3 direction{position.x / distance, position.y / distance, position.z / distance};

            zeroBuffer(resources.inputBuffer);
            const ObjectAudioParams audioParams = ObjectAudioParams::forSlot(effectSlot);
            getAudioSamples(sourceAudio,
                            resources.inputBuffer.data[0],
                            FRAME_SIZE,
                            static_cast<float>(audioTimeSec),
                            SAMPLE_RATE,
                            audioParams.pitchRatio,
                            audioParams.beatOffset,
                            audioParams.beatCycleSec,
                            audioParams.beatDurationSec);

            const float fadeVolume = tracker.getFadeVolume(objectId);
            for (int i = 0; i < FRAME_SIZE; ++i) {
                resources.inputBuffer.data[0][i] *= fadeVolume;
            }

            const IPLfloat32 distanceAttenuation = iplDistanceAttenuationCalculate(
                resources.context, position, LISTENER_POSITION, &resources.distanceModel);
            IPLfloat32 airAbsorption[IPL_NUM_BANDS] = {0.9f, 0.7f, 0.5f};
            iplAirAbsorptionCalculate(resources.context,
                                      position,
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

            iplDirectEffectApply(resources.directEffects[effectSlot],
                                 &directParams,
                                 &resources.inputBuffer,
                                 &resources.inputBuffer);

            zeroBuffer(resources.outputBuffer);
            IPLBinauralEffectParams binauralParams{};
            binauralParams.direction = direction;
            binauralParams.interpolation = IPL_HRTFINTERPOLATION_BILINEAR;
            binauralParams.spatialBlend = 1.0f;
            binauralParams.hrtf = resources.hrtf;

            iplBinauralEffectApply(resources.binauralEffects[effectSlot],
                                   &binauralParams,
                                   &resources.inputBuffer,
                                   &resources.outputBuffer);

            for (int i = 0; i < FRAME_SIZE; ++i) {
                resources.mixBuffer.data[0][i] += resources.outputBuffer.data[0][i];
                resources.mixBuffer.data[1][i] += resources.outputBuffer.data[1][i];
            }
        }

        interleaved.assign(static_cast<size_t>(chunkSamples) * 2, 0.0f);
        for (int i = 0; i < chunkSamples; ++i) {
            interleaved[2 * i] = resources.mixBuffer.data[0][i];
            interleaved[2 * i + 1] = resources.mixBuffer.data[1][i];
        }

        const PaError err = Pa_WriteStream(stream, interleaved.data(), chunkSamples);
        if (err != paNoError && err != paOutputUnderflowed) {
            std::cerr << fmt::format("Error: Pa_WriteStream failed: {}\n", Pa_GetErrorText(err));
            return false;
        }

        rendered += chunkSamples;
        audioTimeSec += static_cast<double>(chunkSamples) / static_cast<double>(SAMPLE_RATE);
    }

    return true;
}

double computeRenderDurationMs(double currentTimestampMs,
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

} // namespace

int main(int argc, char* argv[]) {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    const CLIOptions options = parseCommandLine(argc, argv);
    if (options.showHelp) {
        printUsage(argv[0]);
        return 0;
    }

    WAVFile sourceAudio;
    if (!readWAV(options.audioPath, sourceAudio)) {
        return 1;
    }

    SteamAudioResources steamAudio;
    if (!initializeSteamAudioResources(options.useDefaultHRTF, steamAudio)) {
        return 1;
    }

    PaStream* stream = nullptr;
    if (!initializeOutputStream(options.deviceIndex, stream)) {
        return 1;
    }

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
    const int receiveTimeoutMs = 100;
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

    ObjectTracker tracker;
    std::unordered_map<int, int> objectToSlot;
    std::array<int, MAX_SIMULTANEOUS_OBJECTS> slotToObject;
    slotToObject.fill(-1);

    bool hasPreviousTimestamp = false;
    double previousTimestampMs = 0.0;
    double audioTimeSec = 0.0;
    uint64_t receivedFrames = 0;

    while (gRunning.load()) {
        zmq_msg_t message;
        zmq_msg_init(&message);

        const int recvResult = zmq_msg_recv(&message, socket, 0);
        if (recvResult == -1) {
            const int errnum = zmq_errno();
            zmq_msg_close(&message);
            if (errnum == EAGAIN || errnum == EINTR) {
                continue;
            }
            std::cerr << fmt::format("Error: socket receive failed: {}\n",
                                     zmq_strerror(errnum));
            break;
        }

        const auto* payload = static_cast<const uint8_t*>(zmq_msg_data(&message));
        const size_t payloadLen = zmq_msg_size(&message);

        SocketFrameData frameData;
        std::string parseError;
        const bool parsed = parseSocketObjectRep(payload, payloadLen, frameData, parseError);
        zmq_msg_close(&message);

        if (!parsed) {
            if (zmq_send(socket, "1", 1, 0) == -1) {
                std::cerr << "Error: failed to send parse failure ack\n";
                break;
            }
            std::cerr << fmt::format("Parse failure: {} (payload {} bytes)\n",
                                     parseError,
                                     payloadLen);
            continue;
        }

        if (zmq_send(socket, "0", 1, 0) == -1) {
            std::cerr << "Error: failed to send success ack\n";
            break;
        }

        const double renderDurationMs =
            computeRenderDurationMs(frameData.timestamp_ms, previousTimestampMs, hasPreviousTimestamp);
        previousTimestampMs = frameData.timestamp_ms;
        hasPreviousTimestamp = true;

        const int renderSamples = std::max(
            1,
            static_cast<int>(std::llround((renderDurationMs / 1000.0) * SAMPLE_RATE)));

        const uint64_t frameTimeUs = static_cast<uint64_t>(
            std::max(0.0, frameData.timestamp_ms * 1000.0));
        tracker.updateFromFrame(frameData, frameTimeUs, static_cast<float>(renderSamples));

        if (!renderAndPlaySamples(tracker,
                                  renderSamples,
                                  audioTimeSec,
                                  sourceAudio,
                                  steamAudio,
                                  stream,
                                  objectToSlot,
                                  slotToObject)) {
            break;
        }

        ++receivedFrames;
        if (receivedFrames % 30 == 0) {
            std::cout << fmt::format("Processed {} frames (latest frame {}, objects: {})\n",
                                     receivedFrames,
                                     frameData.frame_number,
                                     frameData.objects.size());
        }
    }

    std::cout << "Shutting down...\n";

    zmq_close(socket);
    zmq_ctx_term(zmqContext);
    shutdownOutputStream(stream);
    steamAudio.release();
    return 0;
}
