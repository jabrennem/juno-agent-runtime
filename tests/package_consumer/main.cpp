#include <memory>
#include <stop_token>

#include "juno_harness/juno_harness.hpp"

namespace {

class PackageConsumerModel final : public juno::harness::Model {
public:
  juno::harness::Expected<juno::harness::GenerationResponse>
  generate(const juno::harness::GenerationRequest &,
           const juno::harness::EventCallback &, std::stop_token) override {
    return juno::harness::make_unexpected(juno::harness::Error{
        juno::harness::ErrorCode::GenerationFailed, "package consumer smoke test"});
  }
};

} // namespace

int main() {
  auto model = std::make_shared<PackageConsumerModel>();
  juno::harness::Agent agent(model);
  auto conversation = agent.start_conversation();
  return conversation.history().empty() ? 0 : 1;
}
