#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <stop_token>

#include "juno_harness/juno_harness.hpp"
#include "juno_harness/fake_backend.hpp"

TEST_CASE("A fake backend returns a final response") {
  auto backend = std::make_shared<juno::harness::FakeBackend>(
      std::vector<juno::harness::FakeStep>{juno::harness::FakeStep::final("hello")});
  juno::harness::Agent agent(backend);
  auto session = agent.create_session();

  auto result = session.run("hi");

  REQUIRE(result);
  CHECK(result.value().final_text == "hello");
  CHECK(result.value().inference_turns == 1);
  CHECK(session.history().size() == 2);
}

TEST_CASE("The agent executes a tool and continues") {
  auto backend = std::make_shared<juno::harness::FakeBackend>(std::vector<juno::harness::FakeStep>{
      juno::harness::FakeStep::calls({{"call-1", "echo", R"({"text":"hello"})"}}),
      juno::harness::FakeStep::final("done"),
  });
  juno::harness::Agent agent(backend);
  REQUIRE(agent.add_tool({"echo", "Echo input", R"({"type":"object"})"},
                         [](std::string_view arguments) -> juno::harness::Result<std::string> {
                           return std::string(arguments);
                         }));
  auto session = agent.create_session();
  auto result = session.run("test");

  REQUIRE(result);
  CHECK(result.value().final_text == "done");
  REQUIRE(session.history().size() == 4);
  CHECK(session.history()[2].role == juno::harness::Role::Tool);
  CHECK(session.history()[2].content == R"({"text":"hello"})");
}

TEST_CASE("Unknown tools become tool-result messages") {
  auto backend = std::make_shared<juno::harness::FakeBackend>(std::vector<juno::harness::FakeStep>{
      juno::harness::FakeStep::calls({{"call-1", "missing", "{}"}}),
      juno::harness::FakeStep::final("recovered"),
  });
  juno::harness::Agent agent(backend);
  auto session = agent.create_session();
  auto result = session.run("test");

  REQUIRE(result);
  CHECK(session.history()[2].content.find("unknown tool") != std::string::npos);
}

TEST_CASE("Malformed tool arguments do not invoke handlers") {
  auto backend = std::make_shared<juno::harness::FakeBackend>(std::vector<juno::harness::FakeStep>{
      juno::harness::FakeStep::calls({{"call-1", "echo", "not-json"}}),
      juno::harness::FakeStep::final("recovered"),
  });
  bool invoked = false;
  juno::harness::Agent agent(backend);
  REQUIRE(agent.add_tool({"echo", "Echo input", "{}"}, [&](std::string_view) -> juno::harness::Result<std::string> {
    invoked = true;
    return std::string{"{}"};
  }));
  auto session = agent.create_session();
  auto result = session.run("test");

  REQUIRE(result);
  CHECK_FALSE(invoked);
  CHECK(session.history()[2].content.find("not valid JSON") != std::string::npos);
}

TEST_CASE("Iteration limits and callback ordering are observable") {
  auto backend = std::make_shared<juno::harness::FakeBackend>(std::vector<juno::harness::FakeStep>{
      juno::harness::FakeStep::calls({{"call-1", "echo", "{}"}}),
      juno::harness::FakeStep::final("unreachable"),
  });
  juno::harness::Agent agent(backend, {.max_inference_turns = 1});
  REQUIRE(agent.add_tool({"echo", "Echo", "{}"}, [](std::string_view) -> juno::harness::Result<std::string> { return std::string{"{}"}; }));
  std::vector<juno::harness::EventType> events;
  auto result = agent.create_session().run("test", [&](const juno::harness::AgentEvent& event) { events.push_back(event.type); });

  REQUIRE_FALSE(result);
  CHECK(result.error().code == juno::harness::ErrorCode::IterationLimitExceeded);
  REQUIRE(events.size() >= 3);
  CHECK(events[0] == juno::harness::EventType::ToolStarted);
  CHECK(events[1] == juno::harness::EventType::ToolCompleted);
  CHECK(events.back() == juno::harness::EventType::Error);
}

TEST_CASE("Separate sessions retain separate histories") {
  auto backend = std::make_shared<juno::harness::FakeBackend>(std::vector<juno::harness::FakeStep>{
      juno::harness::FakeStep::final("first"), juno::harness::FakeStep::final("second")});
  juno::harness::Agent agent(backend, {.system_prompt = "system"});
  auto one = agent.create_session();
  auto two = agent.create_session();
  REQUIRE(one.run("one"));
  REQUIRE(two.run("two"));
  CHECK(one.history()[1].content == "one");
  CHECK(two.history()[1].content == "two");
}

TEST_CASE("Thrown tool handlers become recoverable tool results") {
  auto backend = std::make_shared<juno::harness::FakeBackend>(std::vector<juno::harness::FakeStep>{
      juno::harness::FakeStep::calls({{"call-1", "explode", "{}"}}),
      juno::harness::FakeStep::final("recovered"),
  });
  juno::harness::Agent agent(backend);
  REQUIRE(agent.add_tool({"explode", "Always fails", "{}"}, [](std::string_view) -> juno::harness::Result<std::string> {
    throw std::runtime_error("expected failure");
  }));

  auto result = agent.create_session().run("test");

  REQUIRE(result);
  CHECK(result.value().messages[2].content.find("tool threw: expected failure") != std::string::npos);
}

TEST_CASE("A stopped run returns cancellation before mutating session history") {
  auto backend = std::make_shared<juno::harness::FakeBackend>(
      std::vector<juno::harness::FakeStep>{juno::harness::FakeStep::final("unreachable")});
  juno::harness::Agent agent(backend);
  auto session = agent.create_session();
  std::stop_source stop_source;
  stop_source.request_stop();

  auto result = session.run("test", {}, stop_source.get_token());

  REQUIRE_FALSE(result);
  CHECK(result.error().code == juno::harness::ErrorCode::Cancelled);
  CHECK(session.history().empty());
}
