/**
 * @file agent.hpp
 * @brief Defines the Agent and Conversation classes for managing conversations
 * with a model.
 */

#pragma once

#include <functional>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "juno_harness/model.hpp"

namespace juno::harness {

using JsonObject = nlohmann::json;

using ToolHandler =
    std::function<Expected<std::string>(std::string_view arguments_json)>;

/** The result returned by a readable JSON-based tool handler. */
struct ToolResult {
  bool success{true};
  std::string content;
  JsonObject data{JsonObject::object()};
};

using JsonToolHandler = std::function<ToolResult(const JsonObject &params)>;

/** A parameter shown to the model when a tool is registered. */
struct ToolParameter {
  std::string name;
  std::string description;
  std::string type;
  bool required{false};
};

/** A tool definition and the handler that executes it. */
struct Tool {
  ToolDefinition definition;
  ToolHandler handler;
  JsonToolHandler json_handler;
};

/** Creates a tool with a simple JSON callback and parameter descriptions. */
Tool createTool(const std::string &name, const std::string &description,
                const std::vector<ToolParameter> &parameters,
                JsonToolHandler handler);

/** Immutable behavior shared by every conversation created from an Agent. */
struct AgentSpec {
  std::string system_prompt;
  GenerationConfig generation;
  ReasoningEffort reasoning_effort{ReasoningEffort::Medium};
  std::size_t max_inference_turns{8};
  std::vector<Tool> tools;
};

/** One independent conversation with an Agent. */
class Conversation {
public:
  // Generates a response to the given user message, invoking the callback for
  // events.
  [[nodiscard]] Expected<RunResult> run(std::string_view user_message,
                                        EventCallback callback = {},
                                        std::stop_token stop_token = {});

  // Returns the history of messages in this conversation.
  [[nodiscard]] const std::vector<Message> &history() const;
  void clear();

private:
  friend class Agent;
  Conversation(std::shared_ptr<Model> model,
               std::shared_ptr<const AgentSpec> spec);

  std::shared_ptr<Model> model_;
  std::shared_ptr<const AgentSpec> spec_;
  std::vector<Message> history_;
};

/** Reusable model and immutable behavior used to create conversations. */
class Agent {
public:
  Agent(std::shared_ptr<Model> model, AgentSpec spec = {});
  [[nodiscard]] Conversation start_conversation() const;
  [[nodiscard]] Conversation createConversation() const;

  Agent &setSystemPrompt(std::string system_prompt);
  Agent &setGenerationConfig(GenerationConfig generation);
  Agent &setTemperature(float temperature);
  Agent &setMaxTokens(std::size_t max_tokens);
  Agent &setReasoningEffort(ReasoningEffort reasoning_effort);
  Agent &setMaxInferenceTurns(std::size_t max_inference_turns);
  Agent &registerTool(Tool tool);

private:
  std::shared_ptr<Model> model_;
  AgentSpec spec_;
};

} // namespace juno::harness
