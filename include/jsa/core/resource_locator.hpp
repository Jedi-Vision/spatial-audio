#pragma once

#include <optional>
#include <string>
#include <vector>

namespace jsa::core {

class ResourceLocator {
public:
    ResourceLocator() = default;

    void setAssetsRoot(std::string assetsRoot);
    std::optional<std::string> assetsRoot() const;

    // Resolve relative paths with precedence:
    // 1) --assets-root
    // 2) JSA_ASSET_ROOT
    // 3) repo-local assets/
    std::optional<std::string> resolveAsset(const std::string& path, std::string& err) const;
    std::vector<std::string> candidateRoots() const;

private:
    std::optional<std::string> assetsRoot_;
};

} // namespace jsa::core
