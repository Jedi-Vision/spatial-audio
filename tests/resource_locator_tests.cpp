#include <jsa/core/resource_locator.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

bool writeDummyAsset(const fs::path& root, const std::string& relPath) {
    const fs::path full = root / relPath;
    fs::create_directories(full.parent_path());
    std::ofstream file(full);
    if (!file.is_open()) {
        return false;
    }
    file << "dummy";
    return true;
}

bool testExplicitRootWins() {
    const fs::path base = fs::temp_directory_path() / "jsa_locator_test_explicit";
    const fs::path explicitRoot = base / "explicit";
    const fs::path envRoot = base / "env";

    fs::remove_all(base);
    fs::create_directories(explicitRoot);
    fs::create_directories(envRoot);
    if (!writeDummyAsset(explicitRoot, "beep_1.wav")) {
        return false;
    }
    if (!writeDummyAsset(envRoot, "beep_1.wav")) {
        return false;
    }

    setenv("JSA_ASSET_ROOT", envRoot.c_str(), 1);

    jsa::core::ResourceLocator locator;
    locator.setAssetsRoot(explicitRoot.string());
    std::string err;
    auto resolved = locator.resolveAsset("beep_1.wav", err);
    if (!expect(resolved.has_value(), "explicit root resolves asset")) {
        fs::remove_all(base);
        return false;
    }

    const std::string expected = fs::weakly_canonical(explicitRoot / "beep_1.wav").string();
    const bool ok = expect(*resolved == expected, "explicit root has highest precedence");
    fs::remove_all(base);
    return ok;
}

bool testEnvRootFallback() {
    const fs::path base = fs::temp_directory_path() / "jsa_locator_test_env";
    const fs::path envRoot = base / "env";

    fs::remove_all(base);
    fs::create_directories(envRoot);
    if (!writeDummyAsset(envRoot, "song.wav")) {
        return false;
    }

    setenv("JSA_ASSET_ROOT", envRoot.c_str(), 1);

    jsa::core::ResourceLocator locator;
    std::string err;
    auto resolved = locator.resolveAsset("song.wav", err);
    if (!expect(resolved.has_value(), "env root resolves asset")) {
        fs::remove_all(base);
        return false;
    }

    const std::string expected = fs::weakly_canonical(envRoot / "song.wav").string();
    const bool ok = expect(*resolved == expected, "env root fallback path is used");
    fs::remove_all(base);
    return ok;
}

bool testMissingAssetError() {
    unsetenv("JSA_ASSET_ROOT");
    jsa::core::ResourceLocator locator;
    std::string err;
    auto resolved = locator.resolveAsset("missing.wav", err);
    if (!expect(!resolved.has_value(), "missing asset should not resolve")) {
        return false;
    }
    return expect(err.find("Attempted:") != std::string::npos,
                  "missing asset error includes attempted paths");
}

} // namespace

int main() {
    bool ok = true;
    ok = testExplicitRootWins() && ok;
    ok = testEnvRootFallback() && ok;
    ok = testMissingAssetError() && ok;
    return ok ? 0 : 1;
}
