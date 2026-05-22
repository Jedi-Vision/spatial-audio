#include <jsa/core/wav_io.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

uint16_t readU16(const std::vector<uint8_t>& data, size_t offset) {
    return static_cast<uint16_t>(data[offset] | (data[offset + 1] << 8));
}

uint32_t readU32(const std::vector<uint8_t>& data, size_t offset) {
    return static_cast<uint32_t>(data[offset] |
                                 (data[offset + 1] << 8) |
                                 (data[offset + 2] << 16) |
                                 (data[offset + 3] << 24));
}

int16_t readI16(const std::vector<uint8_t>& data, size_t offset) {
    return static_cast<int16_t>(readU16(data, offset));
}

std::vector<uint8_t> readFile(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
                                std::istreambuf_iterator<char>());
}

fs::path tempPath(const std::string& name) {
    return fs::temp_directory_path() / name;
}

bool testWritesStereoHeaderAndDataSize() {
    const fs::path path = tempPath("jsa_wav_writer_stereo.wav");
    fs::remove(path);

    jsa::core::StreamingWavWriter writer;
    const std::array<float, 6> samples = {
        0.0f, 0.5f,
        -0.5f, 1.0f,
        -1.0f, 0.25f,
    };

    bool ok = true;
    ok = expect(writer.open(path.string(), 48000, 2), "open stereo wav") && ok;
    ok = expect(writer.writeInterleavedFloat32(samples.data(), 3), "write stereo frames") && ok;
    ok = expect(writer.close(), "close stereo wav") && ok;

    const std::vector<uint8_t> bytes = readFile(path);
    ok = expect(bytes.size() == 44 + 12, "stereo file size") && ok;
    ok = expect(std::string(bytes.begin(), bytes.begin() + 4) == "RIFF", "RIFF id") && ok;
    ok = expect(readU32(bytes, 4) == 36 + 12, "RIFF chunk size") && ok;
    ok = expect(std::string(bytes.begin() + 8, bytes.begin() + 12) == "WAVE", "WAVE id") && ok;
    ok = expect(std::string(bytes.begin() + 12, bytes.begin() + 16) == "fmt ", "fmt id") && ok;
    ok = expect(readU16(bytes, 20) == 1, "PCM format") && ok;
    ok = expect(readU16(bytes, 22) == 2, "stereo channels") && ok;
    ok = expect(readU32(bytes, 24) == 48000, "sample rate") && ok;
    ok = expect(readU16(bytes, 34) == 16, "bits per sample") && ok;
    ok = expect(std::string(bytes.begin() + 36, bytes.begin() + 40) == "data", "data id") && ok;
    ok = expect(readU32(bytes, 40) == 12, "data chunk size") && ok;

    fs::remove(path);
    return ok;
}

bool testWritesMonoAndClipsFloatSamples() {
    const fs::path path = tempPath("jsa_wav_writer_mono_clip.wav");
    fs::remove(path);

    jsa::core::StreamingWavWriter writer;
    const std::array<float, 5> samples = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

    bool ok = true;
    ok = expect(writer.open(path.string(), 44100, 1), "open mono wav") && ok;
    ok = expect(writer.writeInterleavedFloat32(samples.data(), samples.size()), "write mono frames") && ok;
    ok = expect(writer.close(), "close mono wav") && ok;

    const std::vector<uint8_t> bytes = readFile(path);
    ok = expect(readU16(bytes, 22) == 1, "mono channels") && ok;
    ok = expect(readU32(bytes, 24) == 44100, "mono sample rate") && ok;
    ok = expect(readU32(bytes, 40) == 10, "mono data size") && ok;
    ok = expect(readI16(bytes, 44) == -32768, "clip below -1") && ok;
    ok = expect(readI16(bytes, 46) == -32768, "-1 maps to min int16") && ok;
    ok = expect(readI16(bytes, 48) == 0, "zero maps to zero") && ok;
    ok = expect(readI16(bytes, 50) == 32767, "1 maps to max int16") && ok;
    ok = expect(readI16(bytes, 52) == 32767, "clip above 1") && ok;

    fs::remove(path);
    return ok;
}

bool testRejectsInvalidInput() {
    const fs::path path = tempPath("jsa_wav_writer_invalid.wav");
    fs::remove(path);

    bool ok = true;
    jsa::core::StreamingWavWriter writer;
    const float sample = 0.0f;
    ok = expect(!writer.open(path.string(), 0, 1), "reject zero sample rate") && ok;
    ok = expect(!writer.open(path.string(), 44100, 0), "reject zero channel count") && ok;
    ok = expect(writer.open(path.string(), 44100, 2), "open valid writer") && ok;
    ok = expect(!writer.writeInterleavedFloat32(nullptr, 1), "reject null nonzero samples") && ok;
    ok = expect(writer.writeInterleavedFloat32(nullptr, 0), "allow null zero frames") && ok;
    ok = expect(writer.writeInterleavedFloat32(&sample, 0), "allow nonnull zero frames") && ok;
    ok = expect(!writer.writeInterleavedFloat32(&sample, (UINT32_MAX / 4u) + 1u),
                "reject header size overflow") && ok;
    ok = expect(writer.close(), "close after invalid writes") && ok;

    fs::remove(path);
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok = testWritesStereoHeaderAndDataSize() && ok;
    ok = testWritesMonoAndClipsFloatSamples() && ok;
    ok = testRejectsInvalidInput() && ok;
    return ok ? 0 : 1;
}
