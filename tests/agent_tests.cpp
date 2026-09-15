#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <stop_token>

#include "juno_harness/fake_model.hpp"
#include "juno_harness/juno_harness.hpp"

TEST_CASE("A fake model returns a final response") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::final("hello")});
  juno::harness::Agent agent(model);
  auto conversation = agent.start_conversation();

  auto result = conversation.run("hi");

  REQUIRE(result);
  CHECK(result.value().final_text == "hello");
  CHECK(result.value().inference_turns == 1);
  CHECK(conversation.history().size() == 2);
  REQUIRE(model->reasoning_effort_requests().size() == 1);
  CHECK(model->reasoning_effort_requests()[0] ==
        juno::harness::ReasoningEffort::Medium);
}

TEST_CASE("AgentSpec propagates reasoning effort to every inference turn") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::calls({{"call-1", "echo", "{}"}}),
          juno::harness::FakeStep::final("done"),
      });
  juno::harness::Agent agent(
      model,
      {.reasoning_effort = juno::harness::ReasoningEffort::Low,
       .tools = {{{"echo", "Echo input", "{}"},
                  [](std::string_view) -> juno::harness::Expected<std::string> {
                    return std::string{"{}"};
                  }}}});

  auto result = agent.start_conversation().run("test");

  REQUIRE(result);
  REQUIRE(model->reasoning_effort_requests().size() == 2);
  CHECK(model->reasoning_effort_requests()[0] ==
        juno::harness::ReasoningEffort::Low);
  CHECK(model->reasoning_effort_requests()[1] ==
        juno::harness::ReasoningEffort::Low);
}

TEST_CASE("The agent executes a tool and continues") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::calls(
              {{"call-1", "echo", R"({"text":"hello"})"}}),
          juno::harness::FakeStep::final("done"),
      });
  juno::harness::Agent agent(
      model, {.tools = {{{"echo", "Echo input", R"({"type":"object"})"},
                         [](std::string_view arguments)
                             -> juno::harness::Expected<std::string> {
                           return std::string(arguments);
                         }}}});
  auto conversation = agent.start_conversation();
  auto result = conversation.run("test");

  REQUIRE(result);
  CHECK(result.value().final_text == "done");
  REQUIRE(conversation.history().size() == 4);
  CHECK(conversation.history()[2].role == juno::harness::Role::Tool);
  CHECK(conversation.history()[2].content == R"({"text":"hello"})");
}

TEST_CASE("A tool returns its own JSON result") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::calls({{"call-1", "get_labels", "{}"}}),
          juno::harness::FakeStep::final("done"),
      });
  juno::harness::Agent agent(
      model,
      {.tools = {
           {{"get_labels", "Return the available track labels.", "{}"},
            [](std::string_view) -> juno::harness::Expected<std::string> {
              return R"(["drums","bass","guitars","rhythm guitars","lead guitars","vocals"])";
            }}}});

  auto result = agent.start_conversation().run("What labels are available?");

  REQUIRE(result);
  CHECK(result.value().messages[2].role == juno::harness::Role::Tool);
  CHECK(
      result.value().messages[2].content ==
      R"(["drums","bass","guitars","rhythm guitars","lead guitars","vocals"])");
}

TEST_CASE("Tool results preserve their JSON text") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::calls({{"call-1", "labels", "{}"}}),
          juno::harness::FakeStep::final("done"),
      });
  juno::harness::Agent agent(
      model,
      {.tools = {
           {{"labels", "Return labels.", "{}"},
            [](std::string_view) -> juno::harness::Expected<std::string> {
              return R"(["quoted \"label\"","line\nbreak","back\\slash"])";
            }}}});

  auto result = agent.start_conversation().run("Get labels");

  REQUIRE(result);
  CHECK(result.value().messages[2].content ==
        R"(["quoted \"label\"","line\nbreak","back\\slash"])");
}

TEST_CASE("Unknown tools become tool-result messages") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::calls({{"call-1", "missing", "{}"}}),
          juno::harness::FakeStep::final("recovered"),
      });
  juno::harness::Agent agent(model);
  auto conversation = agent.start_conversation();
  auto result = conversation.run("test");

  REQUIRE(result);
  CHECK(conversation.history()[2].content.find("unknown tool") !=
        std::string::npos);
}

TEST_CASE("Tool handlers own argument validation") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::calls({{"call-1", "echo", "not-json"}}),
          juno::harness::FakeStep::final("recovered"),
      });
  bool invoked = false;
  juno::harness::Agent agent(
      model,
      {.tools = {{{"echo", "Echo input", "{}"},
                  [&](std::string_view) -> juno::harness::Expected<std::string> {
                    invoked = true;
                    return std::string{"{}"};
                  }}}});
  auto conversation = agent.start_conversation();
  auto result = conversation.run("test");

  REQUIRE(result);
  CHECK(invoked);
  CHECK(conversation.history()[2].content == "{}");
}

TEST_CASE("Iteration limits and callback ordering are observable") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::calls({{"call-1", "echo", "{}"}}),
          juno::harness::FakeStep::final("unreachable"),
      });
  juno::harness::Agent agent(
      model,
      {.max_inference_turns = 1,
       .tools = {{{"echo", "Echo", "{}"},
                  [](std::string_view) -> juno::harness::Expected<std::string> {
                    return std::string{"{}"};
                  }}}});
  std::vector<juno::harness::EventType> events;
  auto result = agent.start_conversation().run(
      "test", [&](const juno::harness::AgentEvent &event) {
        events.push_back(event.type);
      });

  REQUIRE_FALSE(result);
  CHECK(result.error().code ==
        juno::harness::ErrorCode::IterationLimitExceeded);
  REQUIRE(events.size() >= 3);
  CHECK(events[0] == juno::harness::EventType::ToolStarted);
  CHECK(events[1] == juno::harness::EventType::ToolCompleted);
  CHECK(events.back() == juno::harness::EventType::Error);
}

TEST_CASE("Separate conversations retain separate histories") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::final("first"),
          juno::harness::FakeStep::final("second")});
  juno::harness::Agent agent(model, {.system_prompt = "system"});
  auto one = agent.start_conversation();
  auto two = agent.start_conversation();
  REQUIRE(one.run("one"));
  REQUIRE(two.run("two"));
  CHECK(one.history()[1].content == "one");
  CHECK(two.history()[1].content == "two");
}

TEST_CASE("Thrown tool handlers become recoverable tool results") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::calls({{"call-1", "explode", "{}"}}),
          juno::harness::FakeStep::final("recovered"),
      });
  juno::harness::Agent agent(
      model,
      {.tools = {{{"explode", "Always fails", "{}"},
                  [](std::string_view) -> juno::harness::Expected<std::string> {
                    throw std::runtime_error("expected failure");
                  }}}});

  auto result = agent.start_conversation().run("test");

  REQUIRE(result);
  CHECK(result.value().messages[2].content.find(
            "tool threw: expected failure") != std::string::npos);
}

TEST_CASE(
    "A stopped run returns cancellation before mutating conversation history") {
  auto model = std::make_shared<juno::harness::FakeModel>(
      std::vector<juno::harness::FakeStep>{
          juno::harness::FakeStep::final("unreachable")});
  juno::harness::Agent agent(model);
  auto conversation = agent.start_conversation();
  std::stop_source stop_source;
  stop_source.request_stop();

  auto result = conversation.run("test", {}, stop_source.get_token());

  REQUIRE_FALSE(result);
  CHECK(result.error().code == juno::harness::ErrorCode::Cancelled);
  CHECK(conversation.history().empty());
}
