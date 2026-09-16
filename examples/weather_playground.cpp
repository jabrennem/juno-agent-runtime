#include <iostream>
#include <string>
#include <string_view>

#include "juno_harness/juno_harness.hpp"

namespace {

using namespace juno::harness;
void print_help() {
  std::cout << "Ask for a weather forecast for any city.\n"
            << "Example: What's the weather in Seattle for five days?\n"
            << "Commands: /clear, /history, /help, /quit\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " /path/to/model.gguf\n";
    return 2;
  }

  auto model = LlamaCppModel::create(argv[1]);
  if (!model) {
    std::cerr << "Model setup failed: " << model.error().message << '\n';
    return 1;
  }

  // Create a tool for fetching the weather forecast
  auto weather_forecast_tool = createTool(
    "weather_forecast", 
    "Get weather forecast for a city.",
    {
      {"city", "The name of the city", "string", true},
      {"days", "Number of days for the forecast", "integer", false}
    },
    [](const JsonObject &params) -> ToolResult {
      if (!params.contains("city") || !params["city"].is_string())
        return {false, "weather_forecast requires a string argument named 'city'"};
      const int days = params.value("days", 3);
      if (days < 1)
        return {false, "weather_forecast requires days to be at least 1"};
      const auto city = params["city"].get<std::string>();
      return {
        true,
        "Weather forecast for " + city + " for the next " +
        std::to_string(days) +
        " days: partly cloudy with a high of 72F and a low of 58F."
        " This is simulated demo data."
      };
    }
  );

  // Create an agent that uses the model and the weather forecast tool
  Agent weather_agent(model.value());
  weather_agent
      .setSystemPrompt(
          "You are WeatherAgent. Use the weather_forecast tool when the user "
          "asks about the weather. The tool provides a simulated forecast, so "
          "be clear that it is a demo when relevant. Ask for a city if one is "
          "not provided, and keep responses concise.")
      .setMaxInferenceTurns(6)
      .registerTool(weather_forecast_tool);
  auto conversation = weather_agent.createConversation();

  std::cout << "Weather playground\n";
  print_help();

  for (std::string input;
       std::cout << "[weather] > " && std::getline(std::cin, input);) {
    if (input == "/quit" || input == "/exit")
      break;
    if (input == "/help") {
      print_help();
      continue;
    }
    if (input == "/clear") {
      conversation.clear();
      std::cout << "Weather conversation cleared.\n";
      continue;
    }
    if (input == "/history") {
      std::cout << conversation.history().size() << " messages\n";
      continue;
    }
    if (input.empty())
      continue;

    std::cout << "assistant: ";
    auto result = conversation.run(input, [](const AgentEvent &event) {
      if (event.type == EventType::TextDelta)
        std::cout << event.text << std::flush;
    });
    if (!result)
      std::cerr << "\nRun failed: " << result.error().message << '\n';
    else
      std::cout << '\n';
  }
}
