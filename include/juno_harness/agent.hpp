/**
 * @file agent.hpp
 * @brief Defines the Agent and Conversation classes for managing conversations with a model.
 */

#pragma once

#include <functional>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "juno_harness/model.hpp"

namespace juno::harness {

using ToolHandler = std::function<Expected<std::string>(std::string_view arguments_json)>;

/** A tool definition and the handler that executes it. */
struct Tool {
  ToolDefinition definition;
  ToolHandler handler;
};

/** Immutable behavior shared by every conversation created from an Agent. */
struct AgentSpec {
  std::string system_prompt;
  GenerationConfig generation;
  std::size_t max_inference_turns{8};
  std::vector<Tool> tools;
};

/** One independent conversation with an Agent. */
class Conversation {
public:
  // Generates a response to the given user message, invoking the callback for events.
  [[nodiscard]] Expected<RunResult> run(std::string_view user_message, EventCallback callback = {}, std::stop_token stop_token = {});
  
  // Returns the history of messages in this conversation.
  [[nodiscard]] const std::vector<Message> &history() const;
  void clear();

private:
  friend class Agent;
  Conversation(std::shared_ptr<Model> model, std::shared_ptr<const AgentSpec> spec);

  std::shared_ptr<Model> model_;
  std::shared_ptr<const AgentSpec> spec_;
  std::vector<Message> history_;
};

/** Reusable model and immutable behavior used to create conversations. */
class Agent {
public:
  Agent(std::shared_ptr<Model> model, AgentSpec spec = {});
  [[nodiscard]] Conversation start_conversation() const;

private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<const AgentSpec> spec_;
};

} // namespace juno::harness
