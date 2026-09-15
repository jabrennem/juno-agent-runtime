/** @file fake_backend.hpp
 *  @brief Defines a fake inference backend for testing purposes.
 */

#pragma once

#include <vector>

#include "juno_harness/backend.hpp"

namespace juno::harness {

struct FakeStep {
  enum class Kind { Final, ToolCalls, Failure };

  Kind kind{Kind::Final};
  std::string content;
  std::vector<ToolCall> tool_calls;
  Error error{ErrorCode::GenerationFailed, "scripted backend failure"};

  static FakeStep final(std::string text);
  static FakeStep calls(std::vector<ToolCall> calls, std::string text = {});
  static FakeStep failure(Error error);
};

class FakeBackend final : public InferenceBackend {
 public:
  explicit FakeBackend(std::vector<FakeStep> script);

  Result<GenerationResponse> generate(const GenerationRequest& request,
                                      const EventCallback& callback,
                                      std::stop_token stop_token) override;
  [[nodiscard]] std::size_t remaining_steps() const;

 private:
  std::vector<FakeStep> script_;
  std::size_t next_step_{0};
};

}  // namespace juno::harness
