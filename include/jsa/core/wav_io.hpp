#pragma once

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

} // namespace jsa::core
