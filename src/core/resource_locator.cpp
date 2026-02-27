#include <jsa/core/resource_locator.hpp>

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;

namespace {

std::optional<fs::path> findRepoAssetsRootFromCwd() {
    fs::path current = fs::current_path();
    while (!current.empty()) {
        const fs::path assets = current / "assets";
        const fs::path manifest = assets / "manifest" / "assets.lock.json";
        if (fs::exists(assets) && fs::is_directory(assets)) {
            if (fs::exists(manifest)) {
                return assets;
            }
        }

        if (current == current.root_path()) {
            break;
        }
        current = current.parent_path();
    }
    return std::nullopt;
}

void appendIfValid(std::vector<std::string>& out, const std::string& candidate) {
    if (candidate.empty()) {
        return;
    }

    const fs::path p(candidate);
    if (!fs::exists(p) || !fs::is_directory(p)) {
        return;
    }

    const std::string normalized = fs::weakly_canonical(p).string();
    for (const std::string& existing : out) {
        if (existing == normalized) {
            return;
        }
    }

    out.push_back(normalized);
}

} // namespace

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

std::vector<std::string> ResourceLocator::candidateRoots() const {
    std::vector<std::string> roots;

    if (assetsRoot_.has_value()) {
        appendIfValid(roots, *assetsRoot_);
    }

    if (const char* envRoot = std::getenv("JSA_ASSET_ROOT"); envRoot != nullptr) {
        appendIfValid(roots, envRoot);
    }

    if (const std::optional<fs::path> repoAssets = findRepoAssetsRootFromCwd();
        repoAssets.has_value()) {
        appendIfValid(roots, repoAssets->string());
    }

    return roots;
}

std::optional<std::string> ResourceLocator::resolveAsset(const std::string& path, std::string& err) const {
    err.clear();
    if (path.empty()) {
        err = "Asset path is empty.";
        return std::nullopt;
    }

    const fs::path input(path);
    if (input.is_absolute()) {
        if (fs::exists(input)) {
            return fs::weakly_canonical(input).string();
        }
        err = "Absolute asset path does not exist: " + input.string();
        return std::nullopt;
    }

    std::ostringstream attempted;
    attempted << "<assets-root>/" << input.string();

    for (const std::string& root : candidateRoots()) {
        const fs::path candidate = fs::path(root) / input;
        attempted << ", " << candidate.string();
        if (fs::exists(candidate)) {
            return fs::weakly_canonical(candidate).string();
        }
    }

    err = "Asset not found. Attempted: " + attempted.str();
    return std::nullopt;
}

} // namespace jsa::core
