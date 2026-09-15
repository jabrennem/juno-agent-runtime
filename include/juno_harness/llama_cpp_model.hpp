/**
 * @file llama_cpp_model.hpp
 * @brief Defines the LlamaCppModel class for handling LlamaCpp inference.
 *
 * This file contains the declaration of the LlamaCppModel class, which is responsible for
 * managing inference sessions using the LlamaCpp library. It provides methods to create a model
 * instance, generate text based on requests, and handle events during inference.
 *
 */

#pragma once

#include <memory>
#include <string>

#include "juno_harness/model.hpp"

namespace juno::harness {

/** Configuration for the LlamaCpp model. */
struct LlamaCppConfig {
  std::string model_path;

  // Maximum number of tokens to keep in the context window.
  std::size_t context_size{4096};

  // Number of batches to process in parallel. This can improve throughput but may increase latency.
  std::size_t batch_size{512};
  
  // Number of threads to use for inference. If 0, the default number of threads will be used.
  unsigned int threads{0};

  // Optional override for the chat template used by the model. If empty, the default template will be used.
  std::string chat_template_override;
};

/** Model for handling LlamaCpp inference. */
class LlamaCppModel final : public Model {
 public:
  /** Creates a new LlamaCppModel instance with the given configuration. */
  static Expected<std::shared_ptr<LlamaCppModel>> create(LlamaCppConfig config);

  /** Destroys the LlamaCppModel instance. */
  ~LlamaCppModel() override;

  /** Deleted copy constructor and assignment operator. */
  LlamaCppModel(const LlamaCppModel&) = delete;
  LlamaCppModel& operator=(const LlamaCppModel&) = delete;

  /** Generates text based on the given request, invoking the callback for events. */
  Expected<GenerationResponse> generate(const GenerationRequest& request, const EventCallback& callback, std::stop_token stop_token) override;

 private:
  struct Impl;
  explicit LlamaCppModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace juno::harness
