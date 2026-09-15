/** @file types.hpp
 *  @brief Defines types used in the Juno inference model.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace juno::harness {

/** Represents the role of a message in a conversation. */
enum class Role { System, User, Assistant, Tool };

/** Represents a call to a tool during inference. */
struct ToolCall {
  std::string id;
  std::string name;
  std::string arguments_json{"{}"};
};

/** Represents a message in a conversation. */
struct Message {
  Role role;
  std::string content;
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
  std::string tool_name;
};

/** Represents the definition of a tool that can be used during inference. */
struct ToolDefinition {
  std::string name;
  std::string description;
  std::string parameters_json{"{\"type\":\"object\"}"};
};

/** Configuration for text generation during inference. */
struct GenerationConfig {
  std::size_t max_tokens{512};
  float temperature{0.7F};
  float top_p{0.95F};
  unsigned int seed{0};
};

/** Represents the type of an event during inference. */
enum class EventType { TextDelta, ToolStarted, ToolCompleted, Completed, Error };

/** Represents an event during inference. */
struct AgentEvent {
  EventType type;
  std::string text;
  ToolCall tool_call;
};

/** Callback type for handling events during inference. */
using EventCallback = std::function<void(const AgentEvent&)>;

/** Represents the result of a run in the Juno inference model. */
struct RunResult {
  std::string final_text;
  std::vector<Message> messages;
  std::size_t inference_turns{0};
};

}  // namespace juno::harness
