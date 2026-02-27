#include <jsa/core/resource_locator.hpp>

#include <utility>

namespace jsa::core {

void ResourceLocator::setAssetsRoot(std::string assetsRoot) {
    if (assetsRoot.empty()) {
        assetsRoot_.reset();
        return;
    }
    assetsRoot_ = std::move(assetsRoot);
}

std::optional<std::string> ResourceLocator::assetsRoot() const {
    return assetsRoot_;
}

} // namespace jsa::core
