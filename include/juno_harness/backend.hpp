/** @file backend.hpp
 *  @brief Defines the interface for the Juno inference backend.
 */

#pragma once

#include <span>
#include <stop_token>
#include "juno_harness/result.hpp"
#include "juno_harness/types.hpp"

/** Namespace for the Juno inference backend. */
namespace juno::harness {

/** Request for generating text. */
struct GenerationRequest {
  std::span<const Message> messages;
  std::span<const ToolDefinition> tools;
  GenerationConfig config;
};

/** Response for a generation request. */
struct GenerationResponse {
  std::string content;
  std::vector<ToolCall> tool_calls;
};

/** Base class for all inference backends. */
class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;
  virtual Result<GenerationResponse> generate(const GenerationRequest& request,
                                              const EventCallback& callback,
                                              std::stop_token stop_token) = 0;
};

}  // namespace juno::harness
