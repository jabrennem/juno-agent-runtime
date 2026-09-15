/**
 * @file agent.hpp
 * @brief Defines the Agent class for managing inference sessions and tools.
 *
 * This file contains the declaration of the Agent class, which is responsible for managing
 * inference sessions, tools, and configurations. It provides methods to add tools, create
 * sessions, and run inference with event callbacks.
 *
 */

#pragma once

#include <functional>
#include <memory>
#include <stop_token>
#include <string_view>
#include <vector>

#include "juno_harness/model.hpp"

namespace juno::harness {

using ToolHandler = std::function<Result<std::string>(std::string_view arguments_json)>;

/** Configuration for an agent, including system prompt and generation settings. */
struct AgentConfig {
  std::string system_prompt;
  GenerationConfig generation;
  std::size_t max_inference_turns{8};
};

/** Forward declaration of the AgentState class used internally by the Agent. */
namespace detail { struct AgentState; }

/** Forward declaration of the AgentSession class representing an inference session. */
class AgentSession;

/** Represents an agent that manages inference sessions and tools. */
class Agent {
 public:
  Agent(std::shared_ptr<Model> model, AgentConfig config = {});
  Result<void> add_tool(std::string name, std::string description, std::function<std::vector<std::string>()> handler);
  Result<void> add_tool_with_definition(ToolDefinition definition, ToolHandler handler);
  [[nodiscard]] AgentSession create_session() const;

 private:
  std::shared_ptr<detail::AgentState> state_;
};

/** Represents a session for running inference with an agent. */
class AgentSession {
 public:
  [[nodiscard]] Result<RunResult> run(std::string_view user_message,
                                      EventCallback callback = {},
                                      std::stop_token stop_token = {});
  [[nodiscard]] const std::vector<Message>& history() const;
  void clear_history();

 private:
  friend class Agent;
  explicit AgentSession(std::shared_ptr<detail::AgentState> state);
  std::shared_ptr<detail::AgentState> state_;
  std::vector<Message> history_;
};

}  // namespace juno::harness
