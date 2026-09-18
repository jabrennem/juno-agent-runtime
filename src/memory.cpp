#include "juno_harness/memory.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unordered_set>

namespace juno::harness {
namespace {

class JsonMemoryStore final : public MemoryStore {
public:
  explicit JsonMemoryStore(MemoryStoreOptions options)
      : name_(std::move(options.name)), description_(std::move(options.description)),
        path_(std::move(options.path)) {}

  const std::string &name() const override {
    return name_;
  }

  const std::string &description() const override {
    return description_;
  }

  Expected<void> remember(MemoryEntry entry) override {
    auto entries = load();
    if (!entries)
      return entries.error();
    if (entry.id.empty()) {
      entry.id = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    entries.value().push_back(std::move(entry));
    try {
      const std::filesystem::path path = expand_user(path_);
      if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
      std::ofstream output(path);
      if (!output)
        return Error{ErrorCode::ToolExecutionFailed,
                     "could not open memory store: " + path.string()};
      output << entries_to_json(entries.value()).dump(2) << '\n';
    } catch (const std::exception &error) {
      return Error{ErrorCode::ToolExecutionFailed,
                   std::string{"could not write memory store: "} + error.what()};
    }
    return {};
  }

  Expected<std::vector<MemoryEntry>> recall(std::string_view query,
                                            std::size_t max_results) const override {
    auto entries = load();
    if (!entries)
      return entries.error();
    std::vector<std::pair<int, MemoryEntry>> scored_matches;
    const auto query_terms = terms(query);
    for (auto entry = entries.value().rbegin(); entry != entries.value().rend(); ++entry) {
      if (query.empty()) {
        scored_matches.emplace_back(1, *entry);
        continue;
      }

      const std::string searchable = entry->content + " " + entry->metadata.dump();
      const auto searchable_terms = terms(searchable);
      int score = 0;
      for (const auto &term : query_terms)
        if (searchable_terms.contains(term))
          ++score;
      if (score > 0)
        scored_matches.emplace_back(score, *entry);
    }
    std::stable_sort(scored_matches.begin(),
                     scored_matches.end(),
                     [](const auto &left, const auto &right) { return left.first > right.first; });
    std::vector<MemoryEntry> matches;
    for (const auto &[score, entry] : scored_matches) {
      (void)score;
      matches.push_back(entry);
      if (matches.size() >= max_results)
        break;
    }
    return matches;
  }

private:
  std::string name_;
  std::string description_;
  std::string path_;

  static std::unordered_set<std::string> terms(std::string_view text) {
    std::unordered_set<std::string> result;
    std::string term;
    for (const char character : text) {
      if (std::isalnum(static_cast<unsigned char>(character))) {
        term.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
      } else if (!term.empty()) {
        if (term == "hometown")
          term = "home";
        if (term.size() >= 3)
          result.insert(term);
        term.clear();
      }
    }
    if (term == "hometown")
      term = "home";
    if (term.size() >= 3)
      result.insert(term);
    return result;
  }

  static std::filesystem::path expand_user(const std::string &path) {
    if (path.rfind("~/", 0) != 0)
      return path;
    const char *home = std::getenv("HOME");
    return home ? std::filesystem::path(home) / path.substr(2) : std::filesystem::path(path);
  }

  Expected<std::vector<MemoryEntry>> load() const {
    try {
      const std::filesystem::path path = expand_user(path_);
      if (!std::filesystem::exists(path))
        return std::vector<MemoryEntry>{};
      std::ifstream input(path);
      nlohmann::json json;
      input >> json;
      std::vector<MemoryEntry> entries;
      for (const auto &value : json) {
        entries.push_back({value.value("id", ""),
                           value.value("content", ""),
                           value.value("metadata", nlohmann::json::object())});
      }
      return entries;
    } catch (const std::exception &error) {
      return Error{ErrorCode::ToolExecutionFailed,
                   std::string{"could not read memory store: "} + error.what()};
    }
  }

  static nlohmann::json entries_to_json(const std::vector<MemoryEntry> &entries) {
    nlohmann::json json = nlohmann::json::array();
    for (const auto &entry : entries)
      json.push_back({{"id", entry.id}, {"content", entry.content}, {"metadata", entry.metadata}});
    return json;
  }
};

std::vector<std::shared_ptr<MemoryStore>> selected(const MemoryManagerOptions &options,
                                                   const std::vector<std::string> &names) {
  if (names.empty())
    return options.stores;
  std::vector<std::shared_ptr<MemoryStore>> result;
  for (const auto &store : options.stores)
    if (std::find(names.begin(), names.end(), store->name()) != names.end())
      result.push_back(store);
  return result;
}

} // namespace

MemoryManager::MemoryManager(MemoryManagerOptions options) : options_(std::move(options)) {
  if (options_.searchTool.name.empty())
    options_.searchTool.name = "search_memory";
  if (options_.searchTool.description.empty())
    options_.searchTool.description = "Search durable memory for relevant facts.";
  if (options_.addTool.name.empty())
    options_.addTool.name = "add_memory";
  if (options_.addTool.description.empty())
    options_.addTool.description =
        "Save a durable user fact, preference, or recurring context for future conversations.";
}

MemoryManager &MemoryManager::addStore(std::shared_ptr<MemoryStore> store) {
  options_.stores.push_back(std::move(store));
  return *this;
}

MemoryManager &MemoryManager::setPolicy(MemoryPolicy policy) {
  options_.policy = policy;
  return *this;
}

const MemoryPolicy &MemoryManager::policy() const {
  return options_.policy;
}

const MemoryToolConfig &MemoryManager::searchTool() const {
  return options_.searchTool;
}

const MemoryToolConfig &MemoryManager::addTool() const {
  return options_.addTool;
}

const std::vector<std::shared_ptr<MemoryStore>> &MemoryManager::stores() const {
  return options_.stores;
}

Expected<void> MemoryManager::remember(MemoryEntry entry, const std::vector<std::string> &stores) {
  auto targets = selected(options_, stores);
  for (const auto &store : targets) {
    auto result = store->remember(entry);
    if (!result)
      return result.error();
  }
  return {};
}

Expected<std::vector<MemoryEntry>>
MemoryManager::recall(std::string_view query, const std::vector<std::string> &stores) const {
  if (!options_.policy.auto_recall_enabled)
    return std::vector<MemoryEntry>{};
  std::vector<MemoryEntry> result;
  for (const auto &store : selected(options_, stores)) {
    auto entries = store->recall(query, options_.policy.max_recalled_memories);
    if (!entries)
      return entries.error();
    result.insert(result.end(), entries.value().begin(), entries.value().end());
  }
  if (result.size() > options_.policy.max_recalled_memories)
    result.resize(options_.policy.max_recalled_memories);
  return result;
}

std::shared_ptr<MemoryManager> createMemoryManager(MemoryManagerOptions options) {
  return std::make_shared<MemoryManager>(std::move(options));
}

std::shared_ptr<MemoryStore> createMemoryStore(const std::string &name, const std::string &path) {
  return createMemoryStore(MemoryStoreOptions{.name = name, .path = path});
}

std::shared_ptr<MemoryStore> createMemoryStore(MemoryStoreOptions options) {
  return std::make_shared<JsonMemoryStore>(std::move(options));
}

std::shared_ptr<MemoryStore> createMemoryStore(const std::string &path) {
  const auto filename = std::filesystem::path(path).filename().string();
  const auto name = std::filesystem::path(filename).stem().string();
  return createMemoryStore(name.empty() ? "memory" : name, path);
}

} // namespace juno::harness
