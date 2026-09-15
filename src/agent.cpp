/**
 * @file agent.cpp
 * @brief Implements the Agent class for managing inference sessions and tools.
 *
 * This file contains the implementation of the Agent class, which is responsible for managing
 * inference sessions, tools, and configurations. It provides methods to add tools, create
 * sessions, and run inference with event callbacks.
 *
 */

#include "juno_harness/agent.hpp"

#include <cctype>
#include <exception>
#include <unordered_map>
#include <utility>

namespace juno::harness::detail {

struct RegisteredTool {
  ToolDefinition definition;
  ToolHandler handler;
};

struct AgentState {
  std::shared_ptr<InferenceBackend> backend;
  AgentConfig config;
  std::unordered_map<std::string, RegisteredTool> tools;
};

}  // namespace juno::harness::detail

namespace juno::harness {
namespace {

bool is_plausible_json(std::string_view value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  const auto last = value.find_last_not_of(" \t\r\n");
  if (first == std::string_view::npos || last == std::string_view::npos) return false;
  const char opening = value[first];
  const char closing = value[last];
  if ((opening == '{' && closing == '}') || (opening == '[' && closing == ']') ||
      opening == '"' || std::isdigit(static_cast<unsigned char>(opening)) || opening == '-' ||
      value.substr(first, last - first + 1) == "true" ||
      value.substr(first, last - first + 1) == "false" ||
      value.substr(first, last - first + 1) == "null") {
    return true;
  }
  return false;
}

std::string tool_error_json(const std::string_view message) {
  std::string escaped;
  escaped.reserve(message.size());
  for (const char character : message) {
    if (character == '"' || character == '\\') escaped.push_back('\\');
    escaped.push_back(character);
  }
  return "{\"error\":\"" + escaped + "\"}";
}

void emit(const EventCallback& callback, AgentEvent event) {
  if (callback) callback(event);
}

}  // namespace

Agent::Agent(std::shared_ptr<InferenceBackend> backend, AgentConfig config)
    : state_(std::make_shared<detail::AgentState>(detail::AgentState{std::move(backend), std::move(config), {}})) {}

Result<void> Agent::add_tool(ToolDefinition definition, ToolHandler handler) {
  if (definition.name.empty() || !handler) {
    return Error{ErrorCode::InvalidConfiguration, "tools require a name and a handler"};
  }
  if (!is_plausible_json(definition.parameters_json)) {
    return Error{ErrorCode::InvalidConfiguration, "tool parameters must be JSON"};
  }
  if (state_->tools.contains(definition.name)) {
    return Error{ErrorCode::InvalidConfiguration, "a tool with this name is already registered"};
  }
  const std::string name = definition.name;
  state_->tools.emplace(name, detail::RegisteredTool{std::move(definition), std::move(handler)});
  return {};
}

AgentSession Agent::create_session() const { return AgentSession{state_}; }

AgentSession::AgentSession(std::shared_ptr<detail::AgentState> state) : state_(std::move(state)) {
  if (!state_->config.system_prompt.empty()) {
    history_.push_back(Message{Role::System, state_->config.system_prompt});
  }
}

Result<RunResult> AgentSession::run(const std::string_view user_message, EventCallback callback,
                                    std::stop_token stop_token) {
  if (!state_->backend) {
    return Error{ErrorCode::InvalidConfiguration, "agent has no inference backend"};
  }
  if (stop_token.stop_requested()) {
    return Error{ErrorCode::Cancelled, "agent run was cancelled"};
  }

  history_.push_back(Message{Role::User, std::string(user_message)});
  std::vector<ToolDefinition> definitions;
  definitions.reserve(state_->tools.size());
  for (const auto& [_, registered] : state_->tools) definitions.push_back(registered.definition);

  for (std::size_t turn = 1; turn <= state_->config.max_inference_turns; ++turn) {
    if (stop_token.stop_requested()) {
      emit(callback, AgentEvent{EventType::Error, "agent run was cancelled", {}});
      return Error{ErrorCode::Cancelled, "agent run was cancelled"};
    }
    const GenerationRequest request{history_, definitions, state_->config.generation};
    auto response = state_->backend->generate(request, callback, stop_token);
    if (!response) {
      emit(callback, AgentEvent{EventType::Error, response.error().message, {}});
      return response.error();
    }

    history_.push_back(Message{Role::Assistant, response.value().content, response.value().tool_calls});
    if (response.value().tool_calls.empty()) {
      RunResult result{response.value().content, history_, turn};
      emit(callback, AgentEvent{EventType::Completed, result.final_text, {}});
      return result;
    }

    for (const ToolCall& call : response.value().tool_calls) {
      if (stop_token.stop_requested()) {
        emit(callback, AgentEvent{EventType::Error, "agent run was cancelled", call});
        return Error{ErrorCode::Cancelled, "agent run was cancelled"};
      }
      emit(callback, AgentEvent{EventType::ToolStarted, {}, call});
      std::string output;
      const auto registered = state_->tools.find(call.name);
      if (registered == state_->tools.end()) {
        output = tool_error_json("unknown tool: " + call.name);
      } else if (!is_plausible_json(call.arguments_json)) {
        output = tool_error_json("tool arguments are not valid JSON");
      } else {
        try {
          auto tool_result = registered->second.handler(call.arguments_json);
          output = tool_result ? tool_result.value() : tool_error_json(tool_result.error().message);
        } catch (const std::exception& exception) {
          output = tool_error_json(std::string("tool threw: ") + exception.what());
        } catch (...) {
          output = tool_error_json("tool threw an unknown exception");
        }
      }
      history_.push_back(Message{Role::Tool, std::move(output), {}, call.id, call.name});
      emit(callback, AgentEvent{EventType::ToolCompleted, history_.back().content, call});
    }
  }

  emit(callback, AgentEvent{EventType::Error, "agent reached its inference-turn limit", {}});
  return Error{ErrorCode::IterationLimitExceeded, "agent reached its inference-turn limit"};
}

const std::vector<Message>& AgentSession::history() const { return history_; }

void AgentSession::clear_history() {
  history_.clear();
  if (!state_->config.system_prompt.empty()) {
    history_.push_back(Message{Role::System, state_->config.system_prompt});
  }
}

}  // namespace juno::harness
