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
  std::shared_ptr<Model> model;
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

void append_json_string(std::string& json, const std::string_view value) {
  constexpr char kHexDigits[] = "0123456789abcdef";
  json.push_back('"');
  for (const unsigned char character : value) {
    switch (character) {
      case '"': json += "\\\""; break;
      case '\\': json += "\\\\"; break;
      case '\b': json += "\\b"; break;
      case '\f': json += "\\f"; break;
      case '\n': json += "\\n"; break;
      case '\r': json += "\\r"; break;
      case '\t': json += "\\t"; break;
      default:
        if (character < 0x20) {
          json += "\\u00";
          json.push_back(kHexDigits[character >> 4]);
          json.push_back(kHexDigits[character & 0x0f]);
        } else {
          json.push_back(static_cast<char>(character));
        }
    }
  }
  json.push_back('"');
}

std::string string_vector_json(const std::vector<std::string>& values) {
  std::string json{"["};
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) json.push_back(',');
    append_json_string(json, values[index]);
  }
  json.push_back(']');
  return json;
}

void emit(const EventCallback& callback, AgentEvent event) {
  if (callback) callback(event);
}

}  // namespace

/**
 * @brief Constructs an Agent with the specified model and configuration.
 *
 * This constructor initializes the Agent with a shared pointer to a Model and
 * an optional AgentConfig. The model is used for generating responses, and the configuration
 * allows customization of the agent's behavior.
 *
 * @param model A shared pointer to the Model used for generating responses.
 * @param config An optional AgentConfig for customizing the agent's behavior.
 */
Agent::Agent(std::shared_ptr<Model> model, AgentConfig config)
    : state_(std::make_shared<detail::AgentState>(detail::AgentState{std::move(model), std::move(config), {}})) {}

/**
 * @brief Adds a tool to the agent with the specified definition and handler.
 *
 * This method registers a tool with the agent, allowing it to be called during inference.
 * The tool must have a unique name and a valid handler function. The parameters for the tool
 * must be provided in JSON format.
 *
 * @param definition The ToolDefinition containing the tool's name, description, and parameters.
 * @param handler The ToolHandler function that processes the tool's arguments and returns a result.
 * @return Result<void> indicating success or an Error if the tool could not be added.
 */
Result<void> Agent::add_tool_with_definition(ToolDefinition definition, ToolHandler handler) {
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

/**
 * @brief Adds a tool to the agent with the specified name, description, and handler.
 *
 * This method registers a tool with the agent, allowing it to be called during inference.
 * The tool must have a unique name and a valid handler function. The parameters for the tool
 * are assumed to be an empty JSON object.
 *
 * @param name The unique name of the tool.
 * @param description A brief description of the tool's purpose.
 * @param handler The function that processes the tool's arguments and returns a result.
 * @return Result<void> indicating success or an Error if the tool could not be added.
 */
Result<void> Agent::add_tool(std::string name, std::string description,
                             std::function<std::vector<std::string>()> handler) {
  if (!handler) {
    return Error{ErrorCode::InvalidConfiguration, "tools require a name and a handler"};
  }

  ToolDefinition definition{
      std::move(name), std::move(description),
      R"({"type":"object","properties":{},"additionalProperties":false})"};
  return add_tool_with_definition(
      std::move(definition), [handler = std::move(handler)](std::string_view) -> Result<std::string> {
        return string_vector_json(handler());
      });
}

/**
 * @brief Creates a new AgentSession for interacting with the agent.
 *
 * This method creates a new session that can be used to run inference with the agent.
 * The session maintains its own history and can be used to process user messages and
 * generate responses.
 *
 * @return An AgentSession object for interacting with the agent.
 */
AgentSession Agent::create_session() const { return AgentSession{state_}; }

/**
 * @brief Constructs an AgentSession with the specified shared state.
 *
 * This constructor initializes the AgentSession with a shared pointer to the AgentState,
 * allowing it to access the model, configuration, and registered tools. If a system prompt
 * is configured, it is added to the session's history.
 *
 * @param state A shared pointer to the AgentState containing the model, configuration, and tools.
 */
AgentSession::AgentSession(std::shared_ptr<detail::AgentState> state) : state_(std::move(state)) {
  if (!state_->config.system_prompt.empty()) {
    history_.push_back(Message{Role::System, state_->config.system_prompt});
  }
}

/**
 * @brief Runs the agent session with the given user message and event callback.
 *
 * This method processes the user message, generates a response using the model,
 * and handles tool calls if any are present in the response. It emits events to the provided
 * callback during the process.
 *
 * @param user_message The message from the user to be processed.
 * @param callback The callback function to handle events during the run.
 * @param stop_token A token to request cancellation of the run.
 * @return Result containing RunResult on success or an Error on failure.
 */
Result<RunResult> AgentSession::run(const std::string_view user_message, EventCallback callback,
                                    std::stop_token stop_token) {
  if (!state_->model) {
    return Error{ErrorCode::InvalidConfiguration, "agent has no model"};
  }
  if (stop_token.stop_requested()) {
    return Error{ErrorCode::Cancelled, "agent run was cancelled"};
  }

  // Add the user's message to the session history.
  history_.push_back(Message{Role::User, std::string(user_message)});

  // Prepare the list of tool definitions for the generation request.
  std::vector<ToolDefinition> definitions;
  definitions.reserve(state_->tools.size());
  for (const auto& [_, registered] : state_->tools) definitions.push_back(registered.definition);

  // Run the inference loop for a maximum number of turns as specified in the configuration.
  for (std::size_t turn = 1; turn <= state_->config.max_inference_turns; ++turn) {
    if (stop_token.stop_requested()) {
      emit(callback, AgentEvent{EventType::Error, "agent run was cancelled", {}});
      return Error{ErrorCode::Cancelled, "agent run was cancelled"};
    }

    // Create the generation request with the current history, tool definitions, and generation configuration.
    const GenerationRequest request{history_, definitions, state_->config.generation};
    auto response = state_->model->generate(request, callback, stop_token);
    if (!response) {
      emit(callback, AgentEvent{EventType::Error, response.error().message, {}});
      return response.error();
    }

    // Add the assistant's response to the session history.
    history_.push_back(Message{Role::Assistant, response.value().content, response.value().tool_calls});
    if (response.value().tool_calls.empty()) {
      RunResult result{response.value().content, history_, turn};
      emit(callback, AgentEvent{EventType::Completed, result.final_text, {}});
      return result;
    }

    // Process each tool call in the assistant's response.
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
          // Call the registered tool handler with the provided arguments and capture the output.
          auto tool_result = registered->second.handler(call.arguments_json);
          output = tool_result ? tool_result.value() : tool_error_json(tool_result.error().message);
        } catch (const std::exception& exception) {
          output = tool_error_json(std::string("tool threw: ") + exception.what());
        } catch (...) {
          output = tool_error_json("tool threw an unknown exception");
        }
      }
      // Add the tool's output to the session history and emit a ToolCompleted event.
      history_.push_back(Message{Role::Tool, std::move(output), {}, call.id, call.name});
      emit(callback, AgentEvent{EventType::ToolCompleted, history_.back().content, call});
    }
  }

  // If the maximum number of inference turns is reached, emit an error event and return an error.
  emit(callback, AgentEvent{EventType::Error, "agent reached its inference-turn limit", {}});
  return Error{ErrorCode::IterationLimitExceeded, "agent reached its inference-turn limit"};
}

const std::vector<Message>& AgentSession::history() const { return history_; }

/**
 * @brief Clears the conversation history of the agent session.
 *
 * This method removes all messages from the session's history. If a system prompt is configured,
 * it will be re-added to the history after clearing.
 */
void AgentSession::clear_history() {
  history_.clear();
  if (!state_->config.system_prompt.empty()) {
    history_.push_back(Message{Role::System, state_->config.system_prompt});
  }
}

}  // namespace juno::harness
