#pragma once

#include <optional>
#include <string>

namespace jsa::core {

class ResourceLocator {
public:
    ResourceLocator() = default;

    void setAssetsRoot(std::string assetsRoot);
    std::optional<std::string> assetsRoot() const;

private:
    std::optional<std::string> assetsRoot_;
};

} // namespace jsa::core
