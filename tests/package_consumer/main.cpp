#include <memory>
#include <stop_token>

#include "juno_harness/juno_harness.hpp"

namespace {

class PackageConsumerBackend final : public juno::harness::InferenceBackend {
 public:
  juno::harness::Result<juno::harness::GenerationResponse> generate(const juno::harness::GenerationRequest&,
                                                   const juno::harness::EventCallback&,
                                                   std::stop_token) override {
    return juno::harness::Error{juno::harness::ErrorCode::GenerationFailed, "package consumer smoke test"};
  }
};

}  // namespace

int main() {
  auto backend = std::make_shared<PackageConsumerBackend>();
  juno::harness::Agent agent(backend);
  auto session = agent.create_session();
  return session.history().empty() ? 0 : 1;
}
