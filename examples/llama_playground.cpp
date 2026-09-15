/**
 * @file llama_playground.cpp
 * @brief Example usage of the Juno Harness SDK with the LlamaCpp model.
 *
 * This example demonstrates how to set up a Juno agent using the LlamaCpp model,
 * run an interactive prompt loop, and handle events during inference. It showcases
 * the basic workflow for interacting with the Juno Harness SDK API.
 *
 */

#include <memory>
#include <iostream>
#include <string>
#include "juno_harness/juno_harness.hpp"

int main(int argc, char** argv) {
  // Edit this system prompt and generation configuration, then supply a GGUF path.
  constexpr auto kSystemPrompt = "You are a concise and helpful assistant for a music producer who specializes in mix session preparation. You will be given a list of tracks and their respective stems, and your task is to provide clear and actionable instructions for organizing, labeling, and preparing the session for mixing. Your responses should be structured, easy to follow, and tailored to the needs of a professional mix engineer.";
  constexpr auto kOldUserPrompt = "Given the following list of tracks and their respective stems, please provide detailed instructions for organizing and labeling the session for mixing:\n\n1. Drums: Kick, Snare, Hi-Hat, Toms, Overheads\n2. Bass: Electric Bass, Synth Bass\n3. Guitars: Rhythm Guitar, Lead Guitar\n4. Vocals: Lead Vocal, Backing Vocals\n5. Keys: Piano, Synth Pads\n6. Effects: Reverb, Delay, FX Sounds\n\nPlease ensure that your instructions are clear, concise, and easy to follow for a professional mix engineer.";
  constexpr std::size_t kContextSize = 4096;

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " /path/to/model.gguf\n";
    return 2;
  }

  // Create the LlamaCpp model with the specified model path and context size.
  auto model = juno::harness::LlamaCppModel::create({.model_path = argv[1], .context_size = kContextSize});
  if (!model) {
    std::cerr << "Model setup failed: " << model.error().message << '\n';
    return 1;
  }

  // Create an agent with the LlamaCpp model and the specified system prompt.
  juno::harness::Agent agent(model.value(), {.system_prompt = kSystemPrompt});
  auto session = agent.create_session();

  // Add tools
  agent.add_tool_with_definition({"list_labels", "List all labels a track can belong to.", }, [](std::string_view) {
    return std::vector<std::string>{
        "Drums",
        "Bass",
        "Guitars",
        "Vocals",
        "Keys",
        "Effects"
      };
  });

  // Start the interactive prompt loop.
  std::cout << "Llama playground\n"
            << "Type a message and press Enter. Commands: /help, /clear, /history, /quit\n\n";

  std::string user_message;
  while (true) {
    std::cout << "> " << std::flush;
    if (!std::getline(std::cin, user_message)) {
      std::cout << '\n';
      break;
    }

    if (user_message == "/quit" || user_message == "/exit") break;
    if (user_message == "/help") {
      std::cout << "Commands:\n"
                << "  /help     Show this help\n"
                << "  /clear    Start a new conversation\n"
                << "  /history  Show the number of messages in this conversation\n"
                << "  /quit     Exit the playground\n";
      continue;
    }
    if (user_message == "/clear") {
      session.clear_history();
      std::cout << "Conversation cleared.\n";
      continue;
    }
    if (user_message == "/history") {
      std::cout << "Conversation contains " << session.history().size() << " messages.\n";
      continue;
    }
    if (user_message.empty()) continue;

    // Run the user message through the agent session and handle events.
    std::cout << "assistant: ";
    auto result = session.run(user_message, [](const juno::harness::AgentEvent& event) {
      if (event.type == juno::harness::EventType::TextDelta) std::cout << event.text << std::flush;
    });
    if (!result) {
      std::cerr << "\nRun failed: " << result.error().message << '\n';
      continue;
    }
    std::cout << "\n";
  }

  std::cout << "Goodbye.\n";
  return 0;
}
