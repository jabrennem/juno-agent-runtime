/**
 * @file memory.hpp
 * @brief Memory management interfaces for the Juno SDK.
 *
 * This file defines the interfaces and classes related to memory management
 * within the Juno SDK. It includes definitions for memory entries, memory
 * stores, and the memory manager that orchestrates memory operations.
 *
 * The MemoryManager class provides methods to remember and recall memory entries
 * across multiple memory stores, with configurable policies for recall behavior
 * and agent write permissions.
 *
 * The MemoryStore interface allows for different implementations of memory storage,
 * enabling flexibility in how memories are persisted and retrieved.
 *
 * @note This file is part of the Juno SDK project and is intended for use
 *       within that context. It relies on the nlohmann::json library for JSON
 *       handling and the Expected class for error handling.
 *
 * @see MemoryEntry, MemoryStore, MemoryManager, MemoryPolicy, MemoryManagerOptions
 */
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "juno_sdk/expected.hpp"

namespace juno::sdk {

using MemoryMetadata = nlohmann::json;

// MemoryEntry represents a single memory entry with an ID, content, and optional metadata.
struct MemoryEntry {
  std::string id;
  std::string content;
  MemoryMetadata metadata{MemoryMetadata::object()};
};

struct MemoryStoreOptions {
  std::string name;
  std::string path;
  std::string description;
};

// MemoryStore is an abstract base class that defines the interface for memory storage.
class MemoryStore {
public:
  virtual ~MemoryStore() = default;
  virtual const std::string &name() const = 0;
  virtual const std::string &description() const = 0;
  virtual Expected<void> remember(MemoryEntry entry) = 0;
  virtual Expected<std::vector<MemoryEntry>> recall(std::string_view query,
                                                    std::size_t max_results = 5) const = 0;
};

// MemoryPolicy controls automatic memory recall injected into model context.
struct MemoryPolicy {
  bool auto_recall_enabled{true};
  std::size_t max_recalled_memories{5};
};

// MemoryToolConfig controls whether a memory operation is exposed as an agent tool.
struct MemoryToolConfig {
  bool enabled{false};
  std::string name;
  std::string description;
};

struct MemoryManagerOptions {
  std::vector<std::shared_ptr<MemoryStore>> stores;
  MemoryPolicy policy{};
  MemoryToolConfig search_tool{.enabled = true,
                               .name = "search_memory",
                               .description = "Search durable memory for relevant facts."};
  MemoryToolConfig add_tool{.enabled = false,
                            .name = "add_memory",
                            .description = "Save an important fact for future conversations."};
};

// MemoryManager orchestrates memory operations across multiple memory stores.
class MemoryManager {
public:
  explicit MemoryManager(MemoryManagerOptions options = {});

  MemoryManager &add_store(std::shared_ptr<MemoryStore> store);
  MemoryManager &set_policy(MemoryPolicy policy);
  const MemoryPolicy &policy() const;
  const MemoryToolConfig &search_tool() const;
  const MemoryToolConfig &add_tool() const;
  const std::vector<std::shared_ptr<MemoryStore>> &stores() const;

  Expected<void> remember(MemoryEntry entry, const std::vector<std::string> &stores = {});
  Expected<std::vector<MemoryEntry>> recall(std::string_view query,
                                            const std::vector<std::string> &stores = {}) const;

private:
  MemoryManagerOptions options_;
};

std::shared_ptr<MemoryManager> create_memory_manager(MemoryManagerOptions options = {});
std::shared_ptr<MemoryStore> create_memory_store(MemoryStoreOptions options);
std::shared_ptr<MemoryStore> create_memory_store(const std::string &name, const std::string &path);
std::shared_ptr<MemoryStore> create_memory_store(const std::string &path);


} // namespace juno::sdk
