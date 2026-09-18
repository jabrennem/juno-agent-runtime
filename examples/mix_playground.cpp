#include <iostream>
#include <string>
#include <string_view>

#include "juno_harness/juno_harness.hpp"

namespace {

using namespace juno::harness;

// Returns an AgentSpec configured for the MixAgent.
AgentSpec mix_spec() {
  AgentSpec spec;
  spec.system_prompt = "You are MixAgent. Help a mix engineer through an "
                       "iterative tool loop. Inspect the current "
                       "mix before acting, make incremental and reversible mix "
                       "changes, and explain the intent of "
                       "each operation. Use only mix-scoped operations and "
                       "never claim to hear audio that has not "
                       "been provided. Please be concise and avoid repeating "
                       "yourself. If you are unsure of the next step, ask for guidance.";
  spec.reasoning_effort = ReasoningEffort::High;
  spec.max_inference_turns = 8;
  spec.tools = {
      // Tool for inspecting the current mix state.
      {{"inspect_mix", "Inspect the current mix state.",
        R"({"type":"object","properties":{}})"},
       [](std::string_view) -> Expected<std::string> {
         return R"({"tracks":6,"fader_moves":0,"plugins_added":0,"mix_peak_db":-6.1,"state":"balanced rough mix"})";
       }},
      // Tool for applying a reversible mix operation.
      {{"apply_mix_operation", "Apply one reversible mix operation.",
        R"({"type":"object","properties":{"operation":{"type":"string"}},"required":["operation"]})"},
      [](std::string_view arguments) -> Expected<std::string> {
         return std::string{"{\"accepted\":true,\"request\":"} +
                std::string(arguments) + "}";
       }},
  };
  return spec;
}

void print_help() {
  std::cout << "Ask for an ad-hoc mixing operation.\n"
            << "Example: Inspect the mix and make the next useful reversible "
               "adjustment.\n"
            << "Commands: /clear, /history, /help, /quit\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " /path/to/model.gguf\n";
    return 2;
  }

  // Create a LlamaCppModel instance with the specified model path and context size.
  auto model = createLlamaCppModel({.model_path = argv[1], .context_size = 4096});
  if (!model) {
    std::cerr << "Model setup failed: " << model.error().message << '\n';
    return 1;
  }

  // Create an Agent instance with the MixAgent specification and start a conversation.
  Agent agent(model.value(), mix_spec());
  auto conversation = agent.start_conversation();
  std::cout << "Mix agent playground\n";
  print_help();

  for (std::string input;
       std::cout << "[mix] > " && std::getline(std::cin, input);) {
    if (input == "/quit" || input == "/exit")
      break;
    if (input == "/help") {
      print_help();
      continue;
    }
    if (input == "/clear") {
      conversation.clear();
      std::cout << "Mix conversation cleared.\n";
      continue;
    }
    if (input == "/history") {
      std::cout << conversation.history().size() << " messages\n";
      continue;
    }
    if (input.empty())
      continue;

    std::cout << "Thinking...\n";
    auto result = conversation.run(input, [](const AgentEvent &event) {
      if (event.type == EventType::TextDelta) {
        std::cout << event.text << std::flush;
      } else if (event.type == EventType::ToolStarted) {
        std::cout << "tool: " << event.tool_call.name << "(" << event.tool_call.arguments_json << ")\n";
      } else if (event.type == EventType::ToolCompleted) {
        std::cout << "tool completed: " << event.tool_call.name << "\n";
      } else if (event.type == EventType::Completed) {
        std::cout << "assistant: " << event.text << "\n";
      } else if (event.type == EventType::Error) {
        std::cerr << "error: " << event.text << "\n";
      }
    });
    if (!result)
      std::cerr << "\nRun failed: " << result.error().message << '\n';
    else
      std::cout << '\n';
  }
}
