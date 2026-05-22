#include <jsa/core/wav_io.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint16_t WAV_FORMAT_PCM = 1;
constexpr uint16_t WAV_BITS_PER_SAMPLE = 16;
constexpr uint32_t WAV_MAX_CHUNK_SIZE = std::numeric_limits<uint32_t>::max();

bool readChunkHeader(std::ifstream& file, std::array<char, 4>& chunkId, uint32_t& chunkSize) {
    return static_cast<bool>(file.read(chunkId.data(), 4) &&
                             file.read(reinterpret_cast<char*>(&chunkSize), sizeof(chunkSize)));
}

void writeAscii(std::ofstream& file, const char* text, std::streamsize length) {
    file.write(text, length);
}

void writeU16(std::ofstream& file, uint16_t value) {
    const std::array<char, 2> bytes = {
        static_cast<char>(value & 0xFFu),
        static_cast<char>((value >> 8u) & 0xFFu),
    };
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void writeU32(std::ofstream& file, uint32_t value) {
    const std::array<char, 4> bytes = {
        static_cast<char>(value & 0xFFu),
        static_cast<char>((value >> 8u) & 0xFFu),
        static_cast<char>((value >> 16u) & 0xFFu),
        static_cast<char>((value >> 24u) & 0xFFu),
    };
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

int16_t floatToPcm16(float sample) {
    if (!std::isfinite(sample)) {
        sample = 0.0f;
    }
    const float clipped = std::max(-1.0f, std::min(1.0f, sample));
    if (clipped >= 1.0f) {
        return std::numeric_limits<int16_t>::max();
    }
    if (clipped <= -1.0f) {
        return std::numeric_limits<int16_t>::min();
    }
    return static_cast<int16_t>(std::lrint(clipped * 32767.0f));
}

void writeI16(std::ofstream& file, int16_t value) {
    writeU16(file, static_cast<uint16_t>(value));
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

StreamingWavWriter::~StreamingWavWriter() {
    if (file_.is_open()) {
        close();
    }
}

bool StreamingWavWriter::open(const std::string& path, int sampleRate, int channels) {
    close();
    lastError_.clear();

    if (path.empty()) {
        lastError_ = "WAV output path must not be empty.";
        return false;
    }
    if (sampleRate <= 0) {
        lastError_ = "WAV sample rate must be positive.";
        return false;
    }
    if (channels <= 0 || channels > std::numeric_limits<uint16_t>::max()) {
        lastError_ = "WAV channel count must be in [1, 65535].";
        return false;
    }

    file_.open(path, std::ios::binary | std::ios::trunc);
    if (!file_.is_open()) {
        lastError_ = "Could not open WAV output file: " + path;
        return false;
    }

    sampleRate_ = sampleRate;
    channels_ = channels;
    framesWritten_ = 0;

    const uint16_t blockAlign = static_cast<uint16_t>(
        static_cast<uint32_t>(channels_) * (WAV_BITS_PER_SAMPLE / 8u));
    const uint32_t byteRate = static_cast<uint32_t>(sampleRate_) * blockAlign;

    writeAscii(file_, "RIFF", 4);
    writeU32(file_, 36);
    writeAscii(file_, "WAVE", 4);
    writeAscii(file_, "fmt ", 4);
    writeU32(file_, 16);
    writeU16(file_, WAV_FORMAT_PCM);
    writeU16(file_, static_cast<uint16_t>(channels_));
    writeU32(file_, static_cast<uint32_t>(sampleRate_));
    writeU32(file_, byteRate);
    writeU16(file_, blockAlign);
    writeU16(file_, WAV_BITS_PER_SAMPLE);
    writeAscii(file_, "data", 4);
    writeU32(file_, 0);

    if (!file_.good()) {
        lastError_ = "Failed to write WAV header: " + path;
        file_.close();
        sampleRate_ = 0;
        channels_ = 0;
        framesWritten_ = 0;
        return false;
    }

    return true;
}

bool StreamingWavWriter::writeInterleavedFloat32(const float* samples, uint64_t frameCount) {
    lastError_.clear();
    if (frameCount == 0) {
        return true;
    }
    if (!file_.is_open()) {
        lastError_ = "WAV writer is not open.";
        return false;
    }
    if (samples == nullptr) {
        lastError_ = "Cannot write nonzero WAV frames from a null sample pointer.";
        return false;
    }

    const uint64_t bytesPerFrame = static_cast<uint64_t>(channels_) * sizeof(int16_t);
    if (frameCount > (WAV_MAX_CHUNK_SIZE / bytesPerFrame) ||
        framesWritten_ > ((WAV_MAX_CHUNK_SIZE / bytesPerFrame) - frameCount)) {
        lastError_ = "WAV data size would exceed 32-bit RIFF header limits.";
        return false;
    }

    const uint64_t sampleCount = frameCount * static_cast<uint64_t>(channels_);
    for (uint64_t i = 0; i < sampleCount; ++i) {
        writeI16(file_, floatToPcm16(samples[i]));
    }
    if (!file_.good()) {
        lastError_ = "Failed to write WAV sample data.";
        return false;
    }

    framesWritten_ += frameCount;
    return true;
}

bool StreamingWavWriter::close() {
    lastError_.clear();
    if (!file_.is_open()) {
        return true;
    }

    const uint64_t dataSize64 =
        framesWritten_ * static_cast<uint64_t>(channels_) * sizeof(int16_t);
    if (dataSize64 > WAV_MAX_CHUNK_SIZE || dataSize64 > (WAV_MAX_CHUNK_SIZE - 36u)) {
        lastError_ = "WAV data size exceeds 32-bit RIFF header limits.";
        file_.close();
        sampleRate_ = 0;
        channels_ = 0;
        framesWritten_ = 0;
        return false;
    }

    const uint32_t dataSize = static_cast<uint32_t>(dataSize64);
    const uint32_t riffSize = 36u + dataSize;

    file_.seekp(4, std::ios::beg);
    writeU32(file_, riffSize);
    file_.seekp(40, std::ios::beg);
    writeU32(file_, dataSize);
    file_.close();

    const bool ok = !file_.fail();
    sampleRate_ = 0;
    channels_ = 0;
    if (!ok) {
        lastError_ = "Failed to finalize WAV header.";
        return false;
    }
    return true;
}

bool StreamingWavWriter::isOpen() const {
    return file_.is_open();
}

uint64_t StreamingWavWriter::framesWritten() const {
    return framesWritten_;
}

uint64_t StreamingWavWriter::samplesWritten() const {
    return framesWritten_ * static_cast<uint64_t>(channels_);
}

const std::string& StreamingWavWriter::lastError() const {
    return lastError_;
}

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
