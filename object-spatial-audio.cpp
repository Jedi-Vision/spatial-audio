#include <phonon.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

using json = nlohmann::json;

// Constants
constexpr int MAX_SIMULTANEOUS_OBJECTS = 2;
constexpr int SAMPLE_RATE = 48000;
constexpr int FRAME_SIZE = 1024;
constexpr float VIDEO_FPS = 30.0f;
constexpr float FRAME_INTERVAL_MS = 1000.0f / VIDEO_FPS; // 33.33ms
constexpr float CAMERA_FOV_HORIZONTAL_DEG = 60.0f;
constexpr float CAMERA_FOV_VERTICAL_DEG = 45.0f;
constexpr float FADE_IN_TIME_MS = 100.0f;
constexpr float FADE_OUT_TIME_MS = 100.0f;
constexpr float PULSE_RATE_HZ = 2.0f;
constexpr float BASE_FREQUENCY_HZ = 400.0f;
constexpr float FREQUENCY_STEP_HZ = 100.0f;

constexpr const char* DEFAULT_DETECTION_FILE = "sample_detections.json";
constexpr const char* DEFAULT_OUTPUT_FILE = "output_spatial_objects.wav";
constexpr const char* DEFAULT_HRTF_SOFA = "D2_HRIR_SOFA/D2_44K_16bit_256tap_FIR_SOFA.sofa";
const IPLVector3 LISTENER_POSITION = {0.0f, 0.0f, 0.0f};

// WAV file header structures
struct WAVHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunkSize;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmtSize = 16;
    uint16_t audioFormat = 1; // PCM
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize;
};

struct WAVFile {
    uint32_t sampleRate;
    uint16_t numChannels;
    uint16_t bitsPerSample;
    std::vector<float> samples; // Mono samples as 32-bit float
};

struct CLIOptions {
    std::string detectionsPath = std::string(DEFAULT_DETECTION_FILE);
    std::string outputPath = std::string(DEFAULT_OUTPUT_FILE);
    bool useDefaultHRTF = true;
    bool showHelp = false;
};

struct ObjectAudioConfig {
    int objectId;
    std::string filePath;
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

bool allocateBuffer(IPLContext context, int channels, int numSamples, IPLAudioBuffer& buffer, const std::string& bufferName) {
    buffer.numChannels = channels;
    buffer.numSamples = numSamples;
    buffer.data = nullptr;

    IPLerror error = iplAudioBufferAllocate(context, channels, numSamples, &buffer);
    if (error != IPL_STATUS_SUCCESS || buffer.data == nullptr) {
        std::cerr << fmt::format("Error: Failed to allocate {} buffer\n", bufferName);
        return false;
    }

    return true;
}

// Read WAV file
bool readWAV(const std::string& filename, WAVFile& wav) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    // Validate RIFF header
    if (std::memcmp(header.riff, "RIFF", 4) != 0 || std::memcmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "Error: Not a valid WAV file" << std::endl;
        return false;
    }

    // Handle chunk-based file structure
    if (std::memcmp(header.fmt, "fmt ", 4) != 0) {
        // Skip until we find fmt chunk
        file.seekg(12);
        char chunkID[4];
        uint32_t chunkSize;
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
                    file.seekg(chunkSize - 16, std::ios::cur);
                }
                foundFmt = true;
                break;
            } else {
                file.seekg(chunkSize, std::ios::cur);
            }
        }
        
        if (!foundFmt) {
            std::cerr << "Error: Could not find fmt chunk in WAV file" << std::endl;
            file.close();
            return false;
        }
        
        // Find data chunk (there might be other chunks between fmt and data)
        while (file.read(chunkID, 4) && file.read(reinterpret_cast<char*>(&header.dataSize), 4)) {
            if (std::memcmp(chunkID, "data", 4) == 0) {
                break;
            } else {
                // Skip this chunk
                file.seekg(header.dataSize, std::ios::cur);
            }
        }
        
        if (std::memcmp(chunkID, "data", 4) != 0) {
            std::cerr << "Error: Could not find data chunk in WAV file" << std::endl;
            file.close();
            return false;
        }
    }

    wav.sampleRate = header.sampleRate;
    wav.numChannels = header.numChannels;
    wav.bitsPerSample = header.bitsPerSample;

    if (header.audioFormat != 1) {
        std::cerr << "Error: Only PCM format is supported" << std::endl;
        return false;
    }

    // Read audio data
    const int bytesPerSample = header.bitsPerSample / 8;
    if (bytesPerSample <= 0) {
        std::cerr << fmt::format("Error: Unsupported bits per sample: {}\n", header.bitsPerSample);
        return false;
    }
    
    size_t numSamples = header.dataSize / (bytesPerSample * header.numChannels);
    std::vector<uint8_t> rawData(header.dataSize);
    file.read(reinterpret_cast<char*>(rawData.data()), rawData.size());
    
    if (static_cast<size_t>(file.gcount()) != rawData.size()) {
        std::cerr << "Error: Failed to read WAV data chunk" << std::endl;
        return false;
    }
    
    auto sampleToFloat = [&](const uint8_t* ptr) -> float {
        switch (header.bitsPerSample) {
            case 8: {
                int32_t value = static_cast<int32_t>(*ptr) - 128; // unsigned 8-bit PCM
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
                int32_t value = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
                return static_cast<float>(value) / 2147483648.0f;
            }
            default:
                std::cerr << fmt::format("Error: Unsupported bits per sample: {}\n", header.bitsPerSample);
                return 0.0f;
        }
    };
    
    // Convert to mono float32 (downmix if needed)
    wav.samples.resize(numSamples);
    
    for (size_t i = 0; i < numSamples; i++) {
        float mixedSample = 0.0f;
        for (uint16_t channel = 0; channel < header.numChannels; channel++) {
            size_t offset = (i * header.numChannels + channel) * bytesPerSample;
            mixedSample += sampleToFloat(&rawData[offset]);
        }
        wav.samples[i] = mixedSample / static_cast<float>(header.numChannels);
    }

    file.close();
    std::cout << fmt::format("Loaded WAV: {} Hz, {} channels, {} samples\n", 
                             wav.sampleRate, header.numChannels, numSamples);
    return true;
}

void printUsage(const char* executableName) {
    std::cout << "Usage: " << executableName << " [INPUT_JSON] [OUTPUT_WAV] [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --hrtf <default|custom>     Use default or custom HRTF (default: default)\n";
    std::cout << "  --help, -h                  Show this help message\n";
}

CLIOptions parseCommandLine(int argc, char* argv[]) {
    CLIOptions options;

    if (argc > 1) {
        options.detectionsPath = argv[1];
    }
    if (argc > 2) {
        options.outputPath = argv[2];
    }

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--hrtf" && i + 1 < argc) {
            std::string_view hrtfType(argv[++i]);
            options.useDefaultHRTF = (hrtfType == "default");
        } else if (arg == "--help" || arg == "-h") {
            options.showHelp = true;
        }
    }

    return options;
}

bool loadObjectAudio(const std::vector<ObjectAudioConfig>& configs,
                     std::unordered_map<int, WAVFile>& audioFiles) {
    for (const auto& config : configs) {
        std::cout << fmt::format("Loading audio file for object ID {}: {}\n",
                                 config.objectId, config.filePath);
        WAVFile audioWav;
        if (!readWAV(config.filePath, audioWav)) {
            std::cerr << fmt::format(
                "Error: Failed to load audio file for object ID {}: {}\n",
                config.objectId, config.filePath);
            return false;
        }
        std::cout << fmt::format("  Object ID {}: {} samples, {} Hz\n",
                                 config.objectId, audioWav.samples.size(),
                                 audioWav.sampleRate);
        audioFiles[config.objectId] = std::move(audioWav);
    }

    std::cout << fmt::format("Loaded audio files for {} object IDs\n", audioFiles.size());
    return true;
}

// Data structures for object detection
struct DetectedObject {
    int id;
    float x_2d;      // Normalized [0, 1]
    float y_2d;      // Normalized [0, 1]
    float depth;     // Depth in meters
    float confidence; // Optional confidence score
    std::string class_name; // Optional class name
};

struct DetectionFrame {
    int frame_number;
    float timestamp_ms;
    std::vector<DetectedObject> objects;
};

// Internal tracking state for interpolation
struct TrackedObject {
    int id;
    IPLVector3 current_position;
    IPLVector3 previous_position;
    float current_distance;
    float previous_distance;
    float fade_volume; // 0.0 to 1.0 for fade in/out
    bool active;
    uint64_t last_seen_timestamp_us;
    
    TrackedObject() : id(-1), current_distance(1.0f), previous_distance(1.0f), 
                     fade_volume(0.0f), active(false), last_seen_timestamp_us(0) {
        current_position = {0.0f, 0.0f, -1.0f};
        previous_position = {0.0f, 0.0f, -1.0f};
    }
};

// Convert 2D camera coordinates + depth to 3D Steam Audio coordinates
IPLVector3 convertToWorldSpace(float x_2d, float y_2d, float depth) {
    // Convert normalized coordinates [0, 1] to angles
    // x_2d: 0 = left, 1 = right
    // y_2d: 0 = top, 1 = bottom
    float normalized_x = x_2d - 0.5f; // [-0.5, 0.5]
    float normalized_y = 0.5f - y_2d; // [-0.5, 0.5] (invert Y)
    
    // Convert to radians
    float fov_h_rad = CAMERA_FOV_HORIZONTAL_DEG * M_PI / 180.0f;
    float fov_v_rad = CAMERA_FOV_VERTICAL_DEG * M_PI / 180.0f;
    
    // Calculate angles
    float horizontal_angle = normalized_x * fov_h_rad;
    float vertical_angle = normalized_y * fov_v_rad;
    
    // Convert to 3D position (camera-relative)
    // Steam Audio uses right-handed: x=right, y=up, z=backward
    IPLVector3 position;
    position.x = depth * std::tan(horizontal_angle);
    position.y = depth * std::tan(vertical_angle);
    position.z = -depth; // Forward in camera space, backward in Steam Audio
    
    return position;
}

// Object tracker for managing object lifecycle and interpolation
class ObjectTracker {
private:
    std::unordered_map<int, TrackedObject> tracked_objects;
    float fade_in_samples;
    float fade_out_samples;
    
public:
    ObjectTracker() {
        fade_in_samples = (FADE_IN_TIME_MS / 1000.0f) * SAMPLE_RATE;
        fade_out_samples = (FADE_OUT_TIME_MS / 1000.0f) * SAMPLE_RATE;
    }
    
    void updateFromFrame(const DetectionFrame& frame, uint64_t current_time_us) {
        // Mark all objects as potentially inactive
        for (auto& pair : tracked_objects) {
            pair.second.active = false;
        }
        
        // Update or create tracked objects
        for (const auto& obj : frame.objects) {
            auto it = tracked_objects.find(obj.id);
            if (it != tracked_objects.end()) {
                // Update existing object
                TrackedObject& tracked = it->second;
                tracked.previous_position = tracked.current_position;
                tracked.previous_distance = tracked.current_distance;
                tracked.current_position = convertToWorldSpace(obj.x_2d, obj.y_2d, obj.depth);
                tracked.current_distance = obj.depth;
                tracked.active = true;
                tracked.last_seen_timestamp_us = current_time_us;
                
                // Fade in if just appeared
                if (tracked.fade_volume < 1.0f) {
                    tracked.fade_volume = std::min(1.0f, tracked.fade_volume + 
                        (FRAME_SIZE / fade_in_samples));
                }
            } else {
                // New object
                TrackedObject tracked;
                tracked.id = obj.id;
                tracked.current_position = convertToWorldSpace(obj.x_2d, obj.y_2d, obj.depth);
                tracked.previous_position = tracked.current_position; // Start with same position
                tracked.current_distance = obj.depth;
                tracked.previous_distance = obj.depth;
                tracked.active = true;
                tracked.fade_volume = 0.0f; // Start faded out
                tracked.last_seen_timestamp_us = current_time_us;
                tracked_objects[obj.id] = tracked;
            }
        }
        
        // Handle fade out for disappeared objects
        for (auto& pair : tracked_objects) {
            if (!pair.second.active) {
                TrackedObject& tracked = pair.second;
                tracked.fade_volume = std::max(0.0f, tracked.fade_volume - 
                    (FRAME_SIZE / fade_out_samples));
            }
        }
    }
    
    // Get interpolated positions for current audio frame
    std::vector<std::pair<int, IPLVector3>> getInterpolatedPositions(
        float interpolation_factor) {
        std::vector<std::pair<int, IPLVector3>> result;
        
        for (auto& pair : tracked_objects) {
            TrackedObject& tracked = pair.second;
            
            // Skip if fully faded out
            if (tracked.fade_volume <= 0.0f) {
                continue;
            }
            
            // Linear interpolation between previous and current position
            IPLVector3 interpolated;
            interpolated.x = tracked.previous_position.x + 
                (tracked.current_position.x - tracked.previous_position.x) * interpolation_factor;
            interpolated.y = tracked.previous_position.y + 
                (tracked.current_position.y - tracked.previous_position.y) * interpolation_factor;
            interpolated.z = tracked.previous_position.z + 
                (tracked.current_position.z - tracked.previous_position.z) * interpolation_factor;
            
            result.push_back({tracked.id, interpolated});
        }
        
        // Limit to max simultaneous objects (prioritize by fade volume)
        if (result.size() > MAX_SIMULTANEOUS_OBJECTS) {
            std::sort(result.begin(), result.end(), 
                [this](const auto& a, const auto& b) {
                    return tracked_objects[a.first].fade_volume > 
                           tracked_objects[b.first].fade_volume;
                });
            result.resize(MAX_SIMULTANEOUS_OBJECTS);
        }
        
        return result;
    }
    
    float getFadeVolume(int object_id) const {
        auto it = tracked_objects.find(object_id);
        if (it != tracked_objects.end()) {
            return it->second.fade_volume;
        }
        return 0.0f;
    }
    
    float getDistance(int object_id) const {
        auto it = tracked_objects.find(object_id);
        if (it != tracked_objects.end()) {
            const TrackedObject& tracked = it->second;
            // Interpolate distance
            return tracked.previous_distance + 
                (tracked.current_distance - tracked.previous_distance) * 0.5f;
        }
        return 1.0f;
    }
    
    void updateInterpolationFactor(float factor) {
        // This could be used for more advanced interpolation
        // For now, we use linear interpolation in getInterpolatedPositions
    }
};

// Get audio samples from loaded WAV file for an object
// Handles resampling and looping if needed
void getAudioSamples(const WAVFile& wav, float* output, int num_samples, 
                     float start_time, float target_sample_rate) {
    if (wav.samples.empty()) {
        // No audio loaded, output silence
        for (int i = 0; i < num_samples; i++) {
            output[i] = 0.0f;
        }
        return;
    }
    
    float source_sample_rate = static_cast<float>(wav.sampleRate);
    size_t source_size = wav.samples.size();
    
    if (source_size == 0) {
        for (int i = 0; i < num_samples; i++) {
            output[i] = 0.0f;
        }
        return;
    }
    
    // If sample rates match, use direct indexing (no resampling needed)
    if (std::abs(source_sample_rate - target_sample_rate) < 1.0f) {
        size_t start_index = static_cast<size_t>(start_time * source_sample_rate) % source_size;
        for (int i = 0; i < num_samples; i++) {
            size_t index = (start_index + i) % source_size;
            output[i] = wav.samples[index];
        }
        return;
    }
    
    // Resample from source sample rate to target sample rate
    for (int i = 0; i < num_samples; i++) {
        // Calculate the time in the source audio
        float target_time = start_time + (static_cast<float>(i) / target_sample_rate);
        float source_sample_index = target_time * source_sample_rate;
        
        // Handle wrapping for looping
        float wrapped_index = std::fmod(source_sample_index, static_cast<float>(source_size));
        if (wrapped_index < 0.0f) {
            wrapped_index += source_size;
        }
        
        // Get the two samples for linear interpolation
        size_t source_index = static_cast<size_t>(wrapped_index);
        size_t next_index = (source_index + 1) % source_size;
        
        // Linear interpolation for smoother playback
        float fraction = wrapped_index - std::floor(wrapped_index);
        float sample1 = wav.samples[source_index];
        float sample2 = wav.samples[next_index];
        output[i] = sample1 + (sample2 - sample1) * fraction;
    }
}

// Read detection frames from JSON file
bool readDetectionFrames(const std::string& filename, std::vector<DetectionFrame>& frames) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int line_number = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        if (line.empty() || line[0] == '#') {
            continue; // Skip empty lines and comments
        }
        
        try {
            json j = json::parse(line);
            
            DetectionFrame frame;
            frame.frame_number = j.value("frame", line_number - 1);
            frame.timestamp_ms = j.value("timestamp_ms", frame.frame_number * FRAME_INTERVAL_MS);
            
            if (j.contains("objects") && j["objects"].is_array()) {
                for (const auto& obj_json : j["objects"]) {
                    DetectedObject obj;
                    obj.id = obj_json.value("id", -1);
                    obj.x_2d = obj_json.value("x", 0.5f);
                    obj.y_2d = obj_json.value("y", 0.5f);
                    obj.depth = obj_json.value("depth", 2.0f);
                    obj.confidence = obj_json.value("confidence", 1.0f);
                    obj.class_name = obj_json.value("class", "");
                    
                    if (obj.id >= 0) {
                        frame.objects.push_back(obj);
                    }
                }
            }
            
            frames.push_back(frame);
        } catch (const json::parse_error& e) {
            std::cerr << fmt::format("Warning: Failed to parse line {}: {}\n", 
                                    line_number, e.what());
        }
    }
    
    file.close();
    std::cout << fmt::format("Loaded {} detection frames\n", frames.size());
    return true;
}

// Write WAV file
bool writeWAV(const std::string& filename, const std::vector<float>& leftChannel, 
              const std::vector<float>& rightChannel, uint32_t sampleRate) {
    if (leftChannel.size() != rightChannel.size()) {
        std::cerr << "Error: Channel size mismatch" << std::endl;
        return false;
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create file: " << filename << std::endl;
        return false;
    }

    WAVHeader header;
    header.numChannels = 2;
    header.sampleRate = sampleRate;
    header.bitsPerSample = 16;
    header.byteRate = sampleRate * header.numChannels * header.bitsPerSample / 8;
    header.blockAlign = header.numChannels * header.bitsPerSample / 8;
    header.dataSize = leftChannel.size() * header.numChannels * header.bitsPerSample / 8;
    header.chunkSize = 36 + header.dataSize;

    file.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));

    // Write interleaved stereo data as 16-bit PCM
    const float scale = 32767.0f;
    for (size_t i = 0; i < leftChannel.size(); i++) {
        int16_t left = static_cast<int16_t>(std::max(-1.0f, std::min(1.0f, leftChannel[i])) * scale);
        int16_t right = static_cast<int16_t>(std::max(-1.0f, std::min(1.0f, rightChannel[i])) * scale);
        file.write(reinterpret_cast<const char*>(&left), sizeof(int16_t));
        file.write(reinterpret_cast<const char*>(&right), sizeof(int16_t));
    }

    file.close();
    std::cout << fmt::format("Wrote WAV: {} Hz, stereo, {} samples\n", sampleRate, leftChannel.size());
    return true;
}

float calculateInterpolationFactor(float currentTimeMs,
                                   size_t currentFrameIdx,
                                   const std::vector<DetectionFrame>& frames) {
    if (currentFrameIdx > 0 && currentFrameIdx < frames.size()) {
        float frameStartMs = frames[currentFrameIdx - 1].timestamp_ms;
        float frameEndMs = frames[currentFrameIdx].timestamp_ms;
        if (frameEndMs > frameStartMs) {
            const float factor = (currentTimeMs - frameStartMs) / (frameEndMs - frameStartMs);
            return std::max(0.0f, std::min(1.0f, factor));
        }
    } else if (currentFrameIdx >= frames.size() && !frames.empty()) {
        // After the last frame, hold the final state
        return 1.0f;
    }

    return 0.0f;
}

bool createHRTF(bool useDefaultHRTF, SteamAudioResources& resources) {
    IPLHRTFSettings hrtfSettings{};
    hrtfSettings.volume = 1.0f;
    hrtfSettings.normType = IPL_HRTFNORMTYPE_NONE;

    IPLerror error = IPL_STATUS_SUCCESS;
    if (useDefaultHRTF) {
        hrtfSettings.type = IPL_HRTFTYPE_DEFAULT;
        std::cout << "Using default HRTF\n";
        error = iplHRTFCreate(resources.context, &resources.audioSettings, &hrtfSettings, &resources.hrtf);
    } else {
        hrtfSettings.type = IPL_HRTFTYPE_SOFA;
        std::string sofaFile(DEFAULT_HRTF_SOFA);
        hrtfSettings.sofaFileName = sofaFile.c_str();
        hrtfSettings.sofaData = nullptr;
        hrtfSettings.sofaDataSize = 0;
        std::cout << fmt::format("Loading HRTF from SOFA file: {}\n", sofaFile);
        error = iplHRTFCreate(resources.context, &resources.audioSettings, &hrtfSettings, &resources.hrtf);
    }

    if (error != IPL_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to create HRTF\n";
        return false;
    }

    std::cout << (useDefaultHRTF ? "Default HRTF created\n" : "Custom HRTF created from SOFA file\n");
    return true;
}

bool initializeSteamAudioResources(bool useDefaultHRTF, SteamAudioResources& resources) {
    resources.release();

    IPLContextSettings contextSettings{};
    contextSettings.version = STEAMAUDIO_VERSION;
    contextSettings.simdLevel = IPL_SIMDLEVEL_NEON;

    IPLerror error = iplContextCreate(&contextSettings, &resources.context);
    if (error != IPL_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to create Steam Audio context\n";
        return false;
    }
    std::cout << "Steam Audio context created\n";

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
    IPLBinauralEffectSettings effectSettings{};
    effectSettings.hrtf = resources.hrtf;
    IPLDirectEffectSettings directSettings{};
    directSettings.numChannels = 1;

    for (int i = 0; i < MAX_SIMULTANEOUS_OBJECTS; ++i) {
        error = iplBinauralEffectCreate(resources.context,
                                        &resources.audioSettings,
                                        &effectSettings,
                                        &resources.binauralEffects[i]);
        if (error != IPL_STATUS_SUCCESS) {
            std::cerr << fmt::format("Error: Failed to create binaural effect {}\n", i) << std::endl;
            resources.release();
            return false;
        }

        error = iplDirectEffectCreate(resources.context,
                                      &resources.audioSettings,
                                      &directSettings,
                                      &resources.directEffects[i]);
        if (error != IPL_STATUS_SUCCESS) {
            std::cerr << fmt::format("Error: Failed to create direct effect {}\n", i) << std::endl;
            resources.release();
            return false;
        }
    }

    std::cout << fmt::format("Created {} binaural and direct effects\n", MAX_SIMULTANEOUS_OBJECTS);
    return true;
}

bool processSpatialAudio(const std::vector<DetectionFrame>& detectionFrames,
                         const std::unordered_map<int, WAVFile>& audioFiles,
                         SteamAudioResources& resources,
                         std::vector<float>& leftOutput,
                         std::vector<float>& rightOutput) {
    if (detectionFrames.empty()) {
        std::cerr << "Error: No detection frames found in file\n";
        return false;
    }

    const float totalDurationMs = detectionFrames.back().timestamp_ms + FRAME_INTERVAL_MS;
    const float totalDurationSec = totalDurationMs / 1000.0f;
    const size_t totalSamples = static_cast<size_t>(totalDurationSec * SAMPLE_RATE);

    std::cout << fmt::format("Processing {} frames ({:.2f} seconds, {} samples)\n",
                             detectionFrames.size(), totalDurationSec, totalSamples);

    ObjectTracker tracker;
    std::unordered_map<int, int> objectToEffectSlot;
    int nextAvailableSlot = 0;

    leftOutput.clear();
    rightOutput.clear();
    leftOutput.reserve(totalSamples);
    rightOutput.reserve(totalSamples);

    size_t currentFrameIdx = 0;
    float maxOutputLeft = 0.0f;
    float maxOutputRight = 0.0f;

    std::cout << "Processing audio...\n";
    for (size_t sampleOffset = 0; sampleOffset < totalSamples; sampleOffset += FRAME_SIZE) {
        const float currentTimeSec = static_cast<float>(sampleOffset) / SAMPLE_RATE;
        const float currentTimeMs = currentTimeSec * 1000.0f;

        while (currentFrameIdx < detectionFrames.size() &&
               detectionFrames[currentFrameIdx].timestamp_ms <= currentTimeMs) {
            uint64_t timestampUs = static_cast<uint64_t>(
                detectionFrames[currentFrameIdx].timestamp_ms * 1000.0f);
            tracker.updateFromFrame(detectionFrames[currentFrameIdx], timestampUs);
            currentFrameIdx++;
        }

        const float interpolationFactor = calculateInterpolationFactor(currentTimeMs, currentFrameIdx, detectionFrames);
        const auto activeObjects = tracker.getInterpolatedPositions(interpolationFactor);

        zeroBuffer(resources.mixBuffer);

        for (const auto& entry : activeObjects) {
            const int objectId = entry.first;
            const IPLVector3& position = entry.second;
            int effectSlot = -1;
            auto slotIt = objectToEffectSlot.find(objectId);
            if (slotIt != objectToEffectSlot.end()) {
                effectSlot = slotIt->second;
            } else if (nextAvailableSlot < MAX_SIMULTANEOUS_OBJECTS) {
                effectSlot = nextAvailableSlot++;
                objectToEffectSlot[objectId] = effectSlot;
                iplBinauralEffectReset(resources.binauralEffects[effectSlot]);
                iplDirectEffectReset(resources.directEffects[effectSlot]);
            } else {
                continue;
            }

            float distance = std::sqrt(position.x * position.x +
                                       position.y * position.y +
                                       position.z * position.z);
            distance = std::max(distance, 0.001f);

            IPLVector3 direction{
                position.x / distance,
                position.y / distance,
                position.z / distance};

            zeroBuffer(resources.inputBuffer);

            auto audioIt = audioFiles.find(objectId);
            if (audioIt != audioFiles.end()) {
                getAudioSamples(audioIt->second,
                                resources.inputBuffer.data[0],
                                FRAME_SIZE,
                                currentTimeSec,
                                SAMPLE_RATE);
            }

            const float fadeVolume = tracker.getFadeVolume(objectId);
            for (int i = 0; i < FRAME_SIZE; ++i) {
                resources.inputBuffer.data[0][i] *= fadeVolume;
            }

            const IPLfloat32 distanceAttenuation = iplDistanceAttenuationCalculate(
                resources.context,
                position,
                LISTENER_POSITION,
                &resources.distanceModel);
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

            IPLBinauralEffectParams params{};
            params.direction = direction;
            params.interpolation = IPL_HRTFINTERPOLATION_BILINEAR;
            params.spatialBlend = 1.0f;
            params.hrtf = resources.hrtf;

            iplBinauralEffectApply(resources.binauralEffects[effectSlot],
                                   &params,
                                   &resources.inputBuffer,
                                   &resources.outputBuffer);

            for (int i = 0; i < FRAME_SIZE; ++i) {
                resources.mixBuffer.data[0][i] += resources.outputBuffer.data[0][i];
                resources.mixBuffer.data[1][i] += resources.outputBuffer.data[1][i];
            }
        }

        const size_t frameSize = std::min(static_cast<size_t>(FRAME_SIZE), totalSamples - sampleOffset);
        for (size_t i = 0; i < frameSize; ++i) {
            leftOutput.push_back(resources.mixBuffer.data[0][i]);
            rightOutput.push_back(resources.mixBuffer.data[1][i]);
            maxOutputLeft = std::max(maxOutputLeft, std::abs(resources.mixBuffer.data[0][i]));
            maxOutputRight = std::max(maxOutputRight, std::abs(resources.mixBuffer.data[1][i]));
        }

        if ((sampleOffset / FRAME_SIZE) % 100 == 0) {
            const float progress = (static_cast<float>(sampleOffset) / totalSamples) * 100.0f;
            std::cout << fmt::format("Progress: {:.1f}%\r", progress);
            std::cout.flush();
        }
    }

    std::cout << fmt::format("\nOutput audio peak levels - Left: {:.4f}, Right: {:.4f}\n",
                             maxOutputLeft, maxOutputRight);

    zeroBuffer(resources.outputBuffer);
    for (const auto& entry : objectToEffectSlot) {
        const int slot = entry.second;
        IPLAudioEffectState tailState = iplBinauralEffectGetTail(
            resources.binauralEffects[slot],
            &resources.outputBuffer);
        while (tailState == IPL_AUDIOEFFECTSTATE_TAILREMAINING) {
            for (int i = 0; i < FRAME_SIZE; ++i) {
                leftOutput.push_back(resources.outputBuffer.data[0][i]);
                rightOutput.push_back(resources.outputBuffer.data[1][i]);
            }
            tailState = iplBinauralEffectGetTail(resources.binauralEffects[slot],
                                                 &resources.outputBuffer);
        }
    }

    std::cout << "\nProcessing complete\n";
    return true;
}

int main(int argc, char* argv[]) {
    CLIOptions options = parseCommandLine(argc, argv);
    if (options.showHelp) {
        printUsage(argv[0]);
        return 0;
    }

    std::unordered_map<int, WAVFile> audioFiles;
    const std::vector<ObjectAudioConfig> audioConfigs = {
        {1, "beep_1.wav"},
        {2, "beep_2.wav"},
    };

    if (!loadObjectAudio(audioConfigs, audioFiles)) {
        return 1;
    }

    std::vector<DetectionFrame> detectionFrames;
    if (!readDetectionFrames(options.detectionsPath, detectionFrames)) {
        return 1;
    }

    if (detectionFrames.empty()) {
        std::cerr << "Error: No detection frames found in file\n";
        return 1;
    }

    SteamAudioResources audioResources;
    if (!initializeSteamAudioResources(options.useDefaultHRTF, audioResources)) {
        return 1;
    }

    std::vector<float> leftOutput;
    std::vector<float> rightOutput;
    if (!processSpatialAudio(detectionFrames, audioFiles, audioResources, leftOutput, rightOutput)) {
        return 1;
    }

    if (!writeWAV(options.outputPath, leftOutput, rightOutput, SAMPLE_RATE)) {
        return 1;
    }

    std::cout << fmt::format("Successfully created spatial audio: {}\n", options.outputPath);
    return 0;
}

