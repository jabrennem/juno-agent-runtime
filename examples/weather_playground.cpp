#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include <curl/curl.h>

#include "juno_harness/juno_harness.hpp"

namespace {

using namespace juno::harness;

constexpr std::string_view kCyan = "\033[36m";
constexpr std::string_view kYellow = "\033[33m";
constexpr std::string_view kBlue = "\033[34m";
constexpr std::string_view kMagenta = "\033[35m";
constexpr std::string_view kReset = "\033[0m";

// Define structures to hold location and weather forecast data

struct Location {
  std::string city;
  double latitude;
  double longitude;
  std::string timezone;
};

struct WeatherForecast {
  Location location;
  double current_temperature;
  double high_temperature;
  double low_temperature;
  int weather_code;
};

// Helper function to write the response data from libcurl into a string
size_t write_response(char *data, size_t size, size_t count, void *user_data) {
  auto *response = static_cast<std::string *>(user_data);
  response->append(data, size * count);
  return size * count;
}

// Fetch JSON data from a given URL using libcurl
Expected<JsonObject> get_json(const std::string &url) {
  CURL *curl = curl_easy_init();
  if (!curl)
    return Error{ErrorCode::ToolExecutionFailed, "could not initialize HTTP client"};

  std::string response;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_response);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);

  const CURLcode result = curl_easy_perform(curl);
  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
  curl_easy_cleanup(curl);

  if (result != CURLE_OK)
    return Error{
        ErrorCode::ToolExecutionFailed,
        std::string{"weather request failed: "} + curl_easy_strerror(result)
    };
  if (status < 200 || status >= 300)
    return Error{
        ErrorCode::ToolExecutionFailed, "weather service returned HTTP " + std::to_string(status)
    };

  try {
    return JsonObject::parse(response);
  } catch (const JsonObject::exception &error) {
    return Error{
        ErrorCode::ToolExecutionFailed,
        std::string{"weather service returned invalid JSON: "} + error.what()
    };
  }
}

// Geocode a city name to get its latitude, longitude, and timezone
Expected<Location> geocode_city(const std::string &city) {
  CURL *curl = curl_easy_init();
  if (!curl)
    return Error{ErrorCode::ToolExecutionFailed, "could not initialize HTTP client"};

  char *encoded = curl_easy_escape(curl, city.c_str(), 0);
  if (!encoded) {
    curl_easy_cleanup(curl);
    return Error{ErrorCode::ToolExecutionFailed, "could not encode city name"};
  }
  const std::string url = "https://geocoding-api.open-meteo.com/v1/search?name="
                          + std::string(encoded) + "&count=1&language=en&format=json";
  curl_free(encoded);
  curl_easy_cleanup(curl);

  auto result = get_json(url);
  if (!result)
    return result.error();
  if (!result.value().contains("results") || result.value()["results"].empty())
    return Error{ErrorCode::ToolExecutionFailed, "could not find that city"};

  const auto &match = result.value()["results"].front();
  return Location{
      .city = match.value("name", city),
      .latitude = match.value("latitude", 0.0),
      .longitude = match.value("longitude", 0.0),
      .timezone = match.value("timezone", "auto")
  };
}

// Fetch the weather forecast for a given location and number of days
Expected<WeatherForecast> fetch_forecast(const Location &location, int days) {
  std::ostringstream url;
  url << "https://api.open-meteo.com/v1/forecast?latitude=" << location.latitude
      << "&longitude=" << location.longitude << "&current=temperature_2m,weather_code"
      << "&daily=temperature_2m_max,temperature_2m_min"
      << "&forecast_days=" << days << "&timezone=auto";

  auto result = get_json(url.str());
  if (!result)
    return result.error();
  try {
    const auto &current = result.value().at("current");
    const auto &daily = result.value().at("daily");
    return WeatherForecast{
        .location = location,
        .current_temperature = current.at("temperature_2m"),
        .high_temperature = daily.at("temperature_2m_max").at(0),
        .low_temperature = daily.at("temperature_2m_min").at(0),
        .weather_code = current.at("weather_code")
    };
  } catch (const JsonObject::exception &error) {
    return Error{
        ErrorCode::ToolExecutionFailed,
        std::string{"weather response was missing expected data: "} + error.what()
    };
  }
}

std::string format_forecast(const WeatherForecast &forecast) {
  std::ostringstream output;
  output << "Weather in " << forecast.location.city << " (" << forecast.location.timezone
         << "): currently " << forecast.current_temperature << " degrees, with a high of "
         << forecast.high_temperature << " and a low of " << forecast.low_temperature
         << " today. Weather code " << forecast.weather_code << ". Data from Open-Meteo.";
  return output.str();
}

bool is_blank(const std::string &text) {
  return text.find_first_not_of(" \t\r\n") == std::string::npos;
}

void print_help() {
  std::cout << "Ask for a weather forecast for any city.\n"
            << "Example: What's the weather in Seattle for five days?\n"
            << "Commands: /clear, /history, /help, /quit\n";
}

} // namespace

int main(int argc, char **argv) {

  // Initialize libcurl for HTTP requests
  curl_global_init(CURL_GLOBAL_DEFAULT);
  const auto curl_cleanup = [] { curl_global_cleanup(); };

  // Check command-line arguments
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " /path/to/model.gguf\n";
    curl_cleanup();
    return 2;
  }

  // Load the LLaMA.cpp model
  auto model = createLlamaCppModel({
    .model_path = argv[1]
  });
  if (!model) {
    std::cerr << "Model setup failed: " << model.error().message << '\n';
    curl_cleanup();
    return 1;
  }

  // Create a memory manager for the user's recent weather requests.
  // auto recent_forecasts_store = createMemoryStore({
  //   .name = "weather",
  //   .path = "~/.juno/memory/recent_forecasts.json",
  //   .description = "Recent cities and weather requests.",
  // });
  // auto weather_memory = createMemoryManager({
  //   .stores = {recent_forecasts_store},
  //   .policy = {
  //     .auto_recall_enabled = false,
  //     .max_recalled_memories = 5
  //   },
  //   .searchTool = {
  //     .enabled = true,
  //     .description = "Look up recent forecasts for locations when useful."
  //   },
  //   .addTool = {
  //     .enabled = true,
  //     .description = "Save durable weather preferences or recurring locations for later conversations."
  //   }
  // });

  // Create a tool for fetching the weather forecast
  auto weather_forecast_tool = createTool({
    .name = "weather_forecast",
    .description = "Get weather forecast for a city.",
    .parameters = {
      {"city", "The city to forecast", "string", true},
      {"days", "Number of days for the forecast", "integer", false}
    },
    .handler = [](const JsonObject &params) -> ToolResult {
      // Determine the city to fetch the forecast for
      std::string city;
      if (params.contains("city") && params["city"].is_string())
        city = params["city"].get<std::string>();
      else if (params.contains("city"))
        return {false, "weather_forecast requires city to be a string"};

      if (city.empty())
        return {false, "Please provide a city."};

      // Determine the number of days for the forecast
      const int days = params.value("days", 1);
      if (days < 1 || days > 16)
        return {false, "weather_forecast supports between 1 and 16 days"};

      // Geocode the city to get its location
      auto geocoded = geocode_city(city);
      if (!geocoded)
        return {false, geocoded.error().message};
      const auto location = geocoded.value();

      // Fetch the weather forecast for the location
      auto forecast = fetch_forecast(location, days);
      if (!forecast)
        return {false, forecast.error().message};

      // Return the formatted forecast
      return {true, format_forecast(forecast.value())};
    }}
  );

  // Create an agent that uses the model and the weather forecast tool
  Agent weather_agent(model.value());
  weather_agent
      .setSystemPrompt(
          "You are WeatherAgent. Answer weather questions concisely using the "
          "available tools and durable memory when useful."
      )
      .setMaxInferenceTurns(6)
      .setReasoningEffort(ReasoningEffort::Medium)
      .registerTool(weather_forecast_tool);
  auto conversation = weather_agent.createConversation();

  // Start the interactive playground
  std::cout << "Weather playground\n";
  print_help();

  for (std::string input; std::cout << "[weather] > " && std::getline(std::cin, input);) {
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

    std::cout << "assistant:\n";
    std::string forecast_fallback;
    auto result = conversation.run(input, [&](const AgentEvent &event) {
      if (event.type == EventType::Prompt) {
        std::cout << kMagenta << event.text << kReset << std::flush;
      } else if (event.type == EventType::ReasoningDelta) {
        std::cout << kCyan << event.text << kReset << std::flush;
      } else if (event.type == EventType::TextDelta) {
        std::cout << kBlue << event.text << kReset << std::flush;
      } else if (event.type == EventType::ToolStarted) {
        std::cout << kYellow << "tool: " << event.tool_call.name << "(" << event.tool_call.arguments_json << ")" << kReset << '\n';
      } else if (event.type == EventType::ToolCompleted) {
        std::cout << kYellow << "tool completed: " << event.tool_call.name << kReset << '\n';
      }
    });
    if (!result)
      std::cout << "\nRun failed: " << result.error().message << '\n';
    else {
      if (is_blank(result.value().final_text)) {
        if (!forecast_fallback.empty()) {
          std::cout << forecast_fallback;
        } else {
          std::cout << "Please provide a city so I can check the weather.";
        }
      }
      std::cout << '\n';
    }
  }
  curl_cleanup();
}
