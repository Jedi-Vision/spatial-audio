#include <jsa/core/wav_io.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace {

bool readChunkHeader(std::ifstream& file, std::array<char, 4>& chunkId, uint32_t& chunkSize) {
    return static_cast<bool>(file.read(chunkId.data(), 4) &&
                             file.read(reinterpret_cast<char*>(&chunkSize), sizeof(chunkSize)));
}

float sampleToFloat(const uint8_t* ptr, int bitsPerSample) {
    switch (bitsPerSample) {
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
}

} // namespace

namespace jsa::core {

bool loadWavFile(const std::string& path, WavData& out, std::string& err) {
    out = {};
    err.clear();

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        err = "Could not open WAV file: " + path;
        return false;
    }

    std::array<char, 4> riff{};
    uint32_t riffSize = 0;
    std::array<char, 4> wave{};
    file.read(riff.data(), 4);
    file.read(reinterpret_cast<char*>(&riffSize), sizeof(riffSize));
    file.read(wave.data(), 4);
    if (!file.good() ||
        std::memcmp(riff.data(), "RIFF", 4) != 0 ||
        std::memcmp(wave.data(), "WAVE", 4) != 0) {
        err = "Not a valid RIFF/WAVE file.";
        return false;
    }

    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    uint32_t dataSize = 0;
    std::streampos dataOffset = 0;
    bool foundFmt = false;
    bool foundData = false;

    while (file.good() && !(foundFmt && foundData)) {
        std::array<char, 4> chunkId{};
        uint32_t chunkSize = 0;
        if (!readChunkHeader(file, chunkId, chunkSize)) {
            break;
        }

        if (std::memcmp(chunkId.data(), "fmt ", 4) == 0) {
            if (chunkSize < 16) {
                err = "Invalid fmt chunk size.";
                return false;
            }

            uint32_t byteRate = 0;
            uint16_t blockAlign = 0;
            file.read(reinterpret_cast<char*>(&audioFormat), sizeof(audioFormat));
            file.read(reinterpret_cast<char*>(&numChannels), sizeof(numChannels));
            file.read(reinterpret_cast<char*>(&sampleRate), sizeof(sampleRate));
            file.read(reinterpret_cast<char*>(&byteRate), sizeof(byteRate));
            file.read(reinterpret_cast<char*>(&blockAlign), sizeof(blockAlign));
            file.read(reinterpret_cast<char*>(&bitsPerSample), sizeof(bitsPerSample));
            if (!file.good()) {
                err = "Failed to read fmt chunk.";
                return false;
            }

            if (chunkSize > 16) {
                file.seekg(static_cast<std::streamoff>(chunkSize - 16), std::ios::cur);
            }
            foundFmt = true;
            continue;
        }

        if (std::memcmp(chunkId.data(), "data", 4) == 0) {
            dataOffset = file.tellg();
            dataSize = chunkSize;
            file.seekg(static_cast<std::streamoff>(chunkSize), std::ios::cur);
            foundData = true;
            continue;
        }

        file.seekg(static_cast<std::streamoff>(chunkSize), std::ios::cur);
    }

    if (!foundFmt) {
        err = "Could not find fmt chunk.";
        return false;
    }
    if (!foundData) {
        err = "Could not find data chunk.";
        return false;
    }
    if (audioFormat != 1) {
        err = "Only PCM WAV is supported.";
        return false;
    }
    if (numChannels == 0) {
        err = "WAV channel count cannot be zero.";
        return false;
    }

    const int bytesPerSample = bitsPerSample / 8;
    if (bytesPerSample <= 0) {
        err = "Unsupported bits per sample value.";
        return false;
    }

    if (dataSize > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        err = "WAV data chunk is too large.";
        return false;
    }

    file.clear();
    file.seekg(dataOffset);
    if (!file.good()) {
        err = "Failed to seek to WAV data chunk.";
        return false;
    }

    std::vector<uint8_t> rawData(dataSize);
    file.read(reinterpret_cast<char*>(rawData.data()),
              static_cast<std::streamsize>(rawData.size()));
    if (static_cast<size_t>(file.gcount()) != rawData.size()) {
        err = "Failed to read full WAV data chunk.";
        return false;
    }

    const size_t numFrames = rawData.size() / (static_cast<size_t>(bytesPerSample) * numChannels);
    out.sampleRate = static_cast<int>(sampleRate);
    out.channels = static_cast<int>(numChannels);
    out.bitsPerSample = static_cast<int>(bitsPerSample);
    out.samples.resize(numFrames);

    for (size_t frameIndex = 0; frameIndex < numFrames; ++frameIndex) {
        float mixed = 0.0f;
        for (uint16_t channel = 0; channel < numChannels; ++channel) {
            const size_t offset =
                (frameIndex * static_cast<size_t>(numChannels) + channel) * bytesPerSample;
            mixed += sampleToFloat(&rawData[offset], bitsPerSample);
        }
        out.samples[frameIndex] = mixed / static_cast<float>(numChannels);
    }

    return true;
}

} // namespace jsa::core
