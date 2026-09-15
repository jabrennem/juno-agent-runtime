/**
 * @file llama_cpp_backend.hpp
 * @brief Defines the LlamaCppBackend class for handling LlamaCpp inference.
 *
 * This file contains the declaration of the LlamaCppBackend class, which is responsible for
 * managing inference sessions using the LlamaCpp library. It provides methods to create a backend
 * instance, generate text based on requests, and handle events during inference.
 *
 */

#pragma once

#include <memory>
#include <string>

#include "juno_harness/backend.hpp"

namespace juno::harness {

/** Configuration for the LlamaCpp backend. */
struct LlamaCppConfig {
  std::string model_path;
  std::size_t context_size{4096};
  std::size_t batch_size{512};
  unsigned int threads{0};
  std::string chat_template_override;
};

/** Backend for handling LlamaCpp inference. */
class LlamaCppBackend final : public InferenceBackend {
 public:
  /** Creates a new LlamaCppBackend instance with the given configuration. */
  static Result<std::shared_ptr<LlamaCppBackend>> create(LlamaCppConfig config);

  /** Destroys the LlamaCppBackend instance. */
  ~LlamaCppBackend() override;

  /** Deleted copy constructor and assignment operator. */
  LlamaCppBackend(const LlamaCppBackend&) = delete;
  LlamaCppBackend& operator=(const LlamaCppBackend&) = delete;

  /** Generates text based on the given request, invoking the callback for events. */
  Result<GenerationResponse> generate(const GenerationRequest& request,
                                      const EventCallback& callback,
                                      std::stop_token stop_token) override;

 private:
  struct Impl;
  explicit LlamaCppBackend(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace juno::harness
