/** @file model.hpp
 *  @brief Defines the interface for the Juno inference model.
 */

#pragma once

#include <span>
#include <stop_token>
#include "juno_harness/expected.hpp"
#include "juno_harness/types.hpp"

/** Namespace for the Juno inference model. */
namespace juno::sdk {

/** Request for generating text. */
struct GenerationRequest {
  std::span<const Message> messages;
  std::span<const ToolDefinition> tools;
  GenerationConfig config;
  ReasoningEffort reasoning_effort{ReasoningEffort::Medium};
};

/** Response for a generation request. */
struct GenerationResponse {
  std::string content;
  std::vector<ToolCall> tool_calls;
};

/** Base class for all inference models. */
class Model {
 public:
  virtual ~Model() = default;
  virtual Expected<GenerationResponse> generate(const GenerationRequest& request,
                                                const EventCallback& callback,
                                                std::stop_token stop_token) = 0;
};

}  // namespace juno::sdk
