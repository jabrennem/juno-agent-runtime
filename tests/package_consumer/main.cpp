#include <memory>
#include <stop_token>

#include "juno_harness/juno_harness.hpp"

namespace {

class PackageConsumerModel final : public juno::sdk::Model {
public:
  juno::sdk::Expected<juno::sdk::GenerationResponse>
  generate(const juno::sdk::GenerationRequest &,
           const juno::sdk::EventCallback &, std::stop_token) override {
    return juno::sdk::make_unexpected(juno::sdk::Error{
        juno::sdk::ErrorCode::GenerationFailed, "package consumer smoke test"});
  }
};

} // namespace

int main() {
  auto model = std::make_shared<PackageConsumerModel>();
  auto agent = juno::sdk::Agent::create({.model = model});
  auto conversation = agent.start_conversation();
  return conversation.history().empty() ? 0 : 1;
}
