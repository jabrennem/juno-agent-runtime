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

std::string parameter_schema(const std::vector<ToolParameter> &parameters) {
  JsonObject schema{{"type", "object"}, {"properties", JsonObject::object()}};
  auto &properties = schema["properties"];
  JsonObject required = JsonObject::array();
  for (const auto &parameter : parameters) {
    properties[parameter.name] = {{"type", parameter.type}, {"description", parameter.description}};
    if (parameter.required)
      required.push_back(parameter.name);
  }
  if (!required.empty())
    schema["required"] = std::move(required);
  return schema.dump();
}

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

std::string memory_store_parameter_description(const MemoryManager &memory) {
  std::string description = "Optional target store. Available stores:";
  for (const auto &store : memory.stores()) {
    description += "\n- " + store->name();
    if (!store->description().empty())
      description += ": " + store->description();
  }
  return description;
}

} // namespace

Tool createTool(ToolOptions options) {
  return Tool{ToolDefinition{std::move(options.name),
                             std::move(options.description),
                             parameter_schema(options.parameters)},
              {},
              std::move(options.handler)};
}

Tool createTool(const std::string &name,
                const std::string &description,
                const std::vector<ToolParameter> &parameters,
                JsonToolHandler handler) {
  return createTool(ToolOptions{.name = name,
                                .description = description,
                                .parameters = parameters,
                                .handler = std::move(handler)});
}

Agent::Agent(std::shared_ptr<Model> model, AgentSpec spec)
    : model_(std::move(model)), spec_(std::move(spec)) {}

Conversation Agent::start_conversation() const {
  return Conversation{model_, std::make_shared<const AgentSpec>(spec_), memory_};
}

Conversation Agent::createConversation() const {
  return start_conversation();
}

Agent &Agent::setSystemPrompt(std::string system_prompt) {
  spec_.system_prompt = std::move(system_prompt);
  return *this;
}

Agent &Agent::setGenerationConfig(GenerationConfig generation) {
  spec_.generation = generation;
  return *this;
}

Agent &Agent::setTemperature(const float temperature) {
  spec_.generation.temperature = temperature;
  return *this;
}

Agent &Agent::setMaxTokens(const std::size_t max_tokens) {
  spec_.generation.max_tokens = max_tokens;
  return *this;
}

Agent &Agent::setReasoningEffort(const ReasoningEffort reasoning_effort) {
  spec_.reasoning_effort = reasoning_effort;
  return *this;
}

Agent &Agent::setMaxInferenceTurns(const std::size_t max_inference_turns) {
  spec_.max_inference_turns = max_inference_turns;
  return *this;
}

Agent &Agent::registerTool(Tool tool) {
  spec_.tools.push_back(std::move(tool));
  return *this;
}

Agent &Agent::setMemory(std::shared_ptr<MemoryManager> memory) {
  memory_ = std::move(memory);
  if (memory_ && memory_->searchTool().enabled) {
    const auto &config = memory_->searchTool();
    const bool already_registered =
        std::any_of(spec_.tools.begin(), spec_.tools.end(), [&config](const Tool &tool) {
          return tool.definition.name == config.name;
        });
    if (!already_registered) {
      registerTool(createTool(
          {.name = config.name,
           .description = config.description,
           .parameters = {{"query", "The durable fact or preference to look up", "string", true}},
           .handler = [memory = memory_](const JsonObject &params) -> ToolResult {
             if (!params.contains("query") || !params["query"].is_string())
               return {false, "search_memory requires a string query"};
             auto results = memory->recall(params["query"].get<std::string>());
             if (!results)
               return {false, results.error().message};
             JsonObject output = JsonObject::array();
             for (const auto &entry : results.value()) {
               output.push_back(
                   {{"content", entry.content}, {"metadata", entry.metadata}, {"store", "memory"}});
             }
             return {true, output.dump()};
           }}));
    }
  }
  if (memory_ && memory_->addTool().enabled) {
    const auto &config = memory_->addTool();
    const bool already_registered =
        std::any_of(spec_.tools.begin(), spec_.tools.end(), [&config](const Tool &tool) {
          return tool.definition.name == config.name;
        });
    if (!already_registered) {
      registerTool(createTool(
          {.name = config.name,
           .description = config.description,
           .parameters = {{"content", "The durable fact or preference to remember", "string", true},
                          {"store", memory_store_parameter_description(*memory_), "string", false},
                          {"metadata", "Optional structured metadata", "object", false}},
           .handler = [memory = memory_](const JsonObject &params) -> ToolResult {
             if (!params.contains("content") || !params["content"].is_string())
               return {false, "add_memory requires string content"};

             std::vector<std::string> stores;
             if (params.contains("store")) {
               if (!params["store"].is_string())
                 return {false, "add_memory store must be a string"};
               stores.push_back(params["store"].get<std::string>());
             }

             auto result =
                 memory->remember({.content = params["content"].get<std::string>(),
                                   .metadata = params.value("metadata", JsonObject::object())},
                                  stores);
             if (!result)
               return {false, result.error().message};
             return {true, "Memory saved."};
           }}));
    }
  }
  return *this;
}

Conversation::Conversation(std::shared_ptr<Model> model,
                           std::shared_ptr<const AgentSpec> spec,
                           std::shared_ptr<MemoryManager> memory)
    : model_(std::move(model)), spec_(std::move(spec)), memory_(std::move(memory)) {
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
Expected<RunResult> Conversation::run(const std::string_view user_message,
                                      EventCallback callback,
                                      std::stop_token stop_token) {
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

  // Recall once for this user turn. Tool-driven inference turns should operate
  // on the same memory snapshot instead of rereading and rescoring storage.
  std::string memory_context;
  if (memory_ && memory_->policy().auto_recall_enabled) {
    auto memories = memory_->recall(user_message);
    if (memories && !memories.value().empty()) {
      JsonObject memory_json = JsonObject::array();
      for (const auto &memory : memories.value())
        memory_json.push_back({{"content", memory.content}, {"metadata", memory.metadata}});
      memory_context =
          "Relevant durable memory. Use it when helpful, but do not treat it as a new user "
          "message:\n" + memory_json.dump();
    }
  }

  // Run the inference loop for a maximum number of turns as specified in the
  // configuration.
  for (std::size_t turn = 1; turn <= spec_->max_inference_turns; ++turn) {
    if (stop_token.stop_requested()) {
      emit(callback, AgentEvent{EventType::Error, "agent run was cancelled", {}});
      return make_unexpected(Error{ErrorCode::Cancelled, "agent run was cancelled"});
    }

    // Create the generation request with the current history, tool definitions, memory, and
    // generation configuration.
    std::vector<Message> model_history = history_;
    if (!memory_context.empty()) {
      if (!model_history.empty() && model_history.front().role == Role::System) {
        // Keep a single leading system message. Several chat templates,
        // including Qwen's, reject a second system message even when it is
        // inserted immediately after the first one.
        model_history.front().content += "\n\n" + memory_context;
      } else {
        model_history.insert(model_history.begin(), Message{Role::System, memory_context});
      }
    }
    const GenerationRequest request{model_history,
                                    definitions,
                                    spec_->generation,
                                    spec_->reasoning_effort};
    auto response = model_->generate(request, callback, stop_token);
    if (!response) {
      emit(callback, AgentEvent{EventType::Error, response.error().message, {}});
      return make_unexpected(response.error());
    }

    // Add the assistant's response to the conversation history.
    history_.push_back(
        Message{Role::Assistant, response.value().content, response.value().tool_calls});
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
      const auto registered =
          std::find_if(spec_->tools.begin(), spec_->tools.end(), [&call](const Tool &tool) {
            return tool.definition.name == call.name;
          });
      if (registered == spec_->tools.end()) {
        output = tool_error_json("unknown tool: " + call.name);
      } else if (!registered->handler && !registered->json_handler) {
        output = tool_error_json("tool has no handler: " + call.name);
      } else {
        try {
          if (registered->json_handler) {
            const auto params = JsonObject::parse(call.arguments_json);
            const auto result = registered->json_handler(params);
            output = result.success ? result.content : tool_error_json(result.content);
            if (output.empty() && !result.data.is_null())
              output = result.data.dump();
          } else {
            auto tool_result = registered->handler(call.arguments_json);
            output =
                tool_result ? tool_result.value() : tool_error_json(tool_result.error().message);
          }
        } catch (const std::exception &exception) {
          output = tool_error_json(std::string("tool threw: ") + exception.what());
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
  emit(callback, AgentEvent{EventType::Error, "agent reached its inference-turn limit", {}});
  return make_unexpected(
      Error{ErrorCode::IterationLimitExceeded, "agent reached its inference-turn limit"});
}

const std::vector<Message> &Conversation::history() const {
  return history_;
}

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
