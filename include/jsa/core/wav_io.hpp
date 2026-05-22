#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace jsa::core {

struct WavData {
    int sampleRate = 0;
    int channels = 0;
    int bitsPerSample = 0;
    std::vector<float> samples;
};

bool loadWavFile(const std::string& path, WavData& out, std::string& err);

class StreamingWavWriter {
public:
    StreamingWavWriter() = default;
    ~StreamingWavWriter();

    StreamingWavWriter(const StreamingWavWriter&) = delete;
    StreamingWavWriter& operator=(const StreamingWavWriter&) = delete;

    bool open(const std::string& path, int sampleRate, int channels);
    bool writeInterleavedFloat32(const float* samples, uint64_t frameCount);
    bool close();

    bool isOpen() const;
    uint64_t framesWritten() const;
    uint64_t samplesWritten() const;
    const std::string& lastError() const;

private:
    std::ofstream file_;
    int sampleRate_ = 0;
    int channels_ = 0;
    uint64_t framesWritten_ = 0;
    std::string lastError_;
};

} // namespace jsa::core
