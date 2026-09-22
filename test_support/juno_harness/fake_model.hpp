/** @file fake_model.hpp
 *  @brief Defines a fake inference model for testing purposes.
 */

#pragma once

#include <vector>

#include "juno_harness/model.hpp"

namespace juno::sdk {

struct FakeStep {
  enum class Kind { Final, ToolCalls, Failure };

  Kind kind{Kind::Final};
  std::string content;
  std::vector<ToolCall> tool_calls;
  Error error{ErrorCode::GenerationFailed, "scripted model failure"};

  static FakeStep final(std::string text);
  static FakeStep calls(std::vector<ToolCall> calls, std::string text = {});
  static FakeStep failure(Error error);
};

class FakeModel final : public Model {
 public:
  explicit FakeModel(std::vector<FakeStep> script);

  Expected<GenerationResponse> generate(const GenerationRequest& request,
                                        const EventCallback& callback,
                                        std::stop_token stop_token) override;
  [[nodiscard]] std::size_t remaining_steps() const;
  [[nodiscard]] const std::vector<ReasoningEffort>& reasoning_effort_requests() const;

 private:
  std::vector<FakeStep> script_;
  std::size_t next_step_{0};
  std::vector<ReasoningEffort> reasoning_effort_requests_;
};

}  // namespace juno::sdk
