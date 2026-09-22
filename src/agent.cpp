/**
 * @file agent.cpp
 * @brief Implements the Agent and Conversation inference loop.
 */

#include "juno_sdk/agent.hpp"

#include <algorithm>
#include <exception>
#include <unordered_set>
#include <utility>

namespace juno::sdk {
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

std::string initial_system_context(const AgentOptions &options) {
  std::string context = options.system_prompt;
  for (const auto &document : options.steering.documents) {
    if (!context.empty())
      context += "\n\n";
    context += "Steering document: " + document.name + "\n" + document.content;
  }
  return context;
}

} // namespace

Tool Tool::create(ToolOptions options) {
  if (options.name.empty())
    throw ConfigurationError(Error{ErrorCode::InvalidConfiguration, "tool name is required"});
  if (options.description.empty())
    throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                   "tool description is required: " + options.name});
  if (!options.handler)
    throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                   "tool handler is required: " + options.name});
  std::unordered_set<std::string> names;
  for (const auto &parameter : options.parameters) {
    if (parameter.name.empty() || parameter.type.empty())
      throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                     "tool parameters require a name and type: " + options.name});
    if (!names.insert(parameter.name).second)
      throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                     "tool parameter names must be unique: " + parameter.name});
  }
  return Tool{ToolDefinition{std::move(options.name), std::move(options.description),
                             parameter_schema(options.parameters)}, {},
              std::move(options.handler)};
}

Agent Agent::create(AgentOptions options) {
  if (!options.model)
    throw ConfigurationError(Error{ErrorCode::InvalidConfiguration, "agent model is required"});
  if (options.max_inference_turns == 0)
    throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                   "max_inference_turns must be greater than zero"});
  if (options.generation.max_tokens == 0)
    throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                   "generation.max_tokens must be greater than zero"});
  if (options.generation.temperature < 0.0F || options.generation.temperature > 2.0F)
    throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                   "generation.temperature must be between 0 and 2"});
  if (options.steering.max_bytes == 0)
    throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                   "steering.max_bytes must be greater than zero"});
  std::size_t steering_bytes = 0;
  std::unordered_set<std::string> steering_names;
  for (const auto &document : options.steering.documents) {
    if (document.name.empty())
      throw ConfigurationError(
          Error{ErrorCode::InvalidConfiguration, "steering document name is required"});
    if (!steering_names.insert(document.name).second)
      throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                     "steering document names must be unique: " + document.name});
    if (document.content.size() > options.steering.max_bytes -
                                     std::min(steering_bytes, options.steering.max_bytes))
      throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                     "steering documents exceed the configured byte limit"});
    steering_bytes += document.content.size();
  }
  std::unordered_set<std::string> names;
  for (const auto &tool : options.tools) {
    if (tool.definition.name.empty() || (!tool.handler && !tool.json_handler))
      throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                     "agent tools must have a name and handler"});
    if (!names.insert(tool.definition.name).second)
      throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                     "agent tool names must be unique: " + tool.definition.name});
  }
  return Agent(std::move(options));
}

Agent::Agent(AgentOptions options)
    : model_(options.model), options_(std::move(options)), memory_(options_.memory) {}

Conversation Agent::start_conversation() const {
  return Conversation{model_, std::make_shared<const AgentOptions>(options_), memory_};
}

void Agent::add_tool(Tool tool) {
  if (tool.definition.name.empty() || (!tool.handler && !tool.json_handler))
    throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                   "agent tools must have a name and handler"});
  const auto duplicate = std::any_of(options_.tools.begin(), options_.tools.end(),
                                     [&tool](const Tool &value) {
                                       return value.definition.name == tool.definition.name;
                                     });
  if (duplicate)
    throw ConfigurationError(Error{ErrorCode::InvalidConfiguration,
                                   "agent tool names must be unique: " + tool.definition.name});
  options_.tools.push_back(std::move(tool));
}

void Agent::set_memory(std::shared_ptr<MemoryManager> memory) {
  memory_ = std::move(memory);
  options_.memory = memory_;
  if (memory_ && memory_->search_tool().enabled) {
    const auto &config = memory_->search_tool();
    const bool already_registered =
        std::any_of(options_.tools.begin(), options_.tools.end(), [&config](const Tool &tool) {
          return tool.definition.name == config.name;
        });
    if (!already_registered) {
      auto tool = Tool::create(
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
           }});
      add_tool(std::move(tool));
    }
  }
  if (memory_ && memory_->add_tool().enabled) {
    const auto &config = memory_->add_tool();
    const bool already_registered =
        std::any_of(options_.tools.begin(), options_.tools.end(), [&config](const Tool &tool) {
          return tool.definition.name == config.name;
        });
    if (!already_registered) {
      auto tool = Tool::create(
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
           }});
      add_tool(std::move(tool));
    }
  }
}

Conversation::Conversation(std::shared_ptr<Model> model,
                           std::shared_ptr<const AgentOptions> options,
                           std::shared_ptr<MemoryManager> memory)
    : model_(std::move(model)), options_(std::move(options)), memory_(std::move(memory)) {
  const auto system_context = initial_system_context(*options_);
  if (!system_context.empty()) {
    history_.push_back(Message{Role::System, system_context});
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
  definitions.reserve(options_->tools.size());
  for (const auto &tool : options_->tools)
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
  for (std::size_t turn = 1; turn <= options_->max_inference_turns; ++turn) {
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
                                    options_->generation,
                                    options_->reasoning_effort};
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
          std::find_if(options_->tools.begin(), options_->tools.end(), [&call](const Tool &tool) {
            return tool.definition.name == call.name;
          });
      if (registered == options_->tools.end()) {
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
  const auto system_context = initial_system_context(*options_);
  if (!system_context.empty()) {
    history_.push_back(Message{Role::System, system_context});
  }
}

} // namespace juno::sdk
