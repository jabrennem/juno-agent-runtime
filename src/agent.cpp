/**
 * @file agent.cpp
 * @brief Implements the Agent and Conversation inference loop.
 */

#include "juno_harness/agent.hpp"

#include <algorithm>
#include <exception>
#include <utility>

namespace juno::harness {
namespace {

std::string tool_error_json(const std::string_view message) {
  std::string escaped;
  escaped.reserve(message.size());
  for (const char character : message) {
    if (character == '"' || character == '\\')
      escaped.push_back('\\');
    escaped.push_back(character);
  }
  return "{\"error\":\"" + escaped + "\"}";
}

void emit(const EventCallback &callback, AgentEvent event) {
  if (callback)
    callback(event);
}

} // namespace

Agent::Agent(std::shared_ptr<Model> model, AgentSpec spec)
    : model_(std::move(model)),
      spec_(std::make_shared<const AgentSpec>(std::move(spec))) {}

Conversation Agent::start_conversation() const {
  return Conversation{model_, spec_};
}

Conversation::Conversation(std::shared_ptr<Model> model,
                           std::shared_ptr<const AgentSpec> spec)
    : model_(std::move(model)), spec_(std::move(spec)) {
  if (!spec_->system_prompt.empty()) {
    history_.push_back(Message{Role::System, spec_->system_prompt});
  }
}

/**
 * @brief Runs the conversation with the given user message and event callback.
 *
 * This method processes the user message, generates a response using the model,
 * and handles tool calls if any are present in the response. It emits events to
 * the provided callback during the process.
 *
 * @param user_message The message from the user to be processed.
 * @param callback The callback function to handle events during the run.
 * @param stop_token A token to request cancellation of the run.
 * @return Expected containing RunResult on success or an Error on failure.
 */
Expected<RunResult> Conversation::run(const std::string_view user_message, EventCallback callback, std::stop_token stop_token) {
  if (!model_) {
    return make_unexpected(Error{ErrorCode::InvalidConfiguration, "agent has no model"});
  }
  if (stop_token.stop_requested()) {
    return make_unexpected(Error{ErrorCode::Cancelled, "agent run was cancelled"});
  }

  // Add the user's message to the conversation history.
  history_.push_back(Message{Role::User, std::string(user_message)});

  // Prepare the list of tool definitions for the generation request.
  std::vector<ToolDefinition> definitions;
  definitions.reserve(spec_->tools.size());
  for (const auto &tool : spec_->tools)
    definitions.push_back(tool.definition);

  // Run the inference loop for a maximum number of turns as specified in the
  // configuration.
  for (std::size_t turn = 1; turn <= spec_->max_inference_turns; ++turn) {
    if (stop_token.stop_requested()) {
      emit(callback, AgentEvent{EventType::Error, "agent run was cancelled", {}});
      return make_unexpected(Error{ErrorCode::Cancelled, "agent run was cancelled"});
    }

    // Create the generation request with the current history, tool definitions,
    // and generation configuration.
    const GenerationRequest request{history_, definitions, spec_->generation};
    auto response = model_->generate(request, callback, stop_token);
    if (!response) {
      emit(callback, AgentEvent{EventType::Error, response.error().message, {}});
      return make_unexpected(response.error());
    }

    // Add the assistant's response to the conversation history.
    history_.push_back(Message{Role::Assistant, response.value().content, response.value().tool_calls});
    if (response.value().tool_calls.empty()) {
      RunResult result{response.value().content, history_, turn};
      emit(callback, AgentEvent{EventType::Completed, result.final_text, {}});
      return result;
    }

    // Process each tool call in the assistant's response.
    for (const ToolCall &call : response.value().tool_calls) {
      if (stop_token.stop_requested()) {
        emit(callback, AgentEvent{EventType::Error, "agent run was cancelled", call});
        return make_unexpected(Error{ErrorCode::Cancelled, "agent run was cancelled"});
      }
      emit(callback, AgentEvent{EventType::ToolStarted, {}, call});
      std::string output;
      const auto registered = std::find_if(spec_->tools.begin(), spec_->tools.end(), [&call](const Tool &tool) {
            return tool.definition.name == call.name;
          });
      if (registered == spec_->tools.end()) {
        output = tool_error_json("unknown tool: " + call.name);
      } else if (!registered->handler) {
        output = tool_error_json("tool has no handler: " + call.name);
      } else {
        try {
          auto tool_result = registered->handler(call.arguments_json);
          output = tool_result ? tool_result.value()
                               : tool_error_json(tool_result.error().message);
        } catch (const std::exception &exception) {
          output =
              tool_error_json(std::string("tool threw: ") + exception.what());
        } catch (...) {
          output = tool_error_json("tool threw an unknown exception");
        }
      }
      // Add the tool's output to the conversation history and emit a
      // ToolCompleted event.
      history_.push_back(Message{Role::Tool, std::move(output), {}, call.id, call.name});
      emit(callback, AgentEvent{EventType::ToolCompleted, history_.back().content, call});
    }
  }

  // If the maximum number of inference turns is reached, emit an error event
  // and return an error.
  emit(callback, AgentEvent{EventType::Error,
                            "agent reached its inference-turn limit",
                            {}});
  return make_unexpected(Error{ErrorCode::IterationLimitExceeded,
                               "agent reached its inference-turn limit"});
}

const std::vector<Message> &Conversation::history() const { return history_; }

/**
 * @brief Clears the conversation history.
 *
 * This method removes all messages from the conversation's history. If a system
 * prompt is configured, it will be re-added to the history after clearing.
 */
void Conversation::clear() {
  history_.clear();
  if (!spec_->system_prompt.empty()) {
    history_.push_back(Message{Role::System, spec_->system_prompt});
  }
}

} // namespace juno::harness
