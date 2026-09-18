# Repository Guide for Coding Agents

## Project overview

Juno Harness is a C++20 SDK for embedding a small tool-using agent loop in
native applications. The public API lives under `include/juno_harness`, the
runtime implementation lives in `src`, and llama.cpp is an optional private
implementation dependency. Keep the SDK usable without exposing llama.cpp in
public headers. The JSON type used by the readable tool API is an intentional
public dependency.

## Build and test

Use the repository helper for normal validation:

```sh
./build.sh
```

For a faster model-free test build, use:

```sh
JUNO_HARNESS_BUILD_DIR=build-test ./build.sh \
  -DJUNO_HARNESS_ENABLE_LLAMA_CPP=OFF
```

Useful environment variables are:

- `JUNO_HARNESS_BUILD_DIR` to select a build directory.
- `JUNO_HARNESS_BUILD_TYPE` to select the CMake build type.
- `JUNO_HARNESS_SKIP_TESTS=1` to compile without running CTest.
- `CMAKE_BIN` and `CTEST_BIN` when those tools are not on `PATH`.

When changing installation, exported targets, or public headers, also verify a
consumer can use the installed package through `find_package(JunoHarness)`.

## Code organization

- `include/juno_harness`: public SDK headers. Avoid leaking private model or
  third-party implementation details here; intentionally public dependencies
  must be documented and exported correctly.
- `src/agent.cpp`: conversation state, the inference loop, tool dispatch, and
  event delivery.
- `src/llama_cpp_model.cpp`: the in-process llama.cpp adapter.
- `test_support`: deterministic `FakeModel` support used only by tests.
- `tests/agent_tests.cpp`: Catch2 coverage of public agent behavior.
- `examples`: standalone playground applications; do not make core SDK behavior
  depend on them.
- `cmake`: installed package configuration.

## Implementation conventions

- Preserve C++20 compatibility and the `juno::harness` namespace.
- Use RAII and value semantics. Prefer standard-library types in public APIs.
- Report expected failures with `Expected<T>` and `Error`; reserve exceptions
  for truly exceptional boundaries. Tool-handler exceptions must remain
  recoverable by the conversation loop.
- Keep `Agent` reusable. Each conversation receives an immutable snapshot of
  the agent configuration, owns an independent transcript, and is not
  thread-safe or reentrant.
- Preserve synchronous callback ordering and emit observable terminal errors.
- Tool arguments and results are JSON text. Handlers own argument validation;
  the harness should not silently rewrite their payloads.
- Keep dependency linkage private unless a type is intentionally part of the
  installed API.
- Match the formatting and naming already used in neighboring code. Avoid
  drive-by formatting or unrelated cleanup.

## Public API design

The primary API should be approachable to developers coming from Python or
other higher-level languages. Prefer a small, object-oriented, setter-based
facade with obvious names and simple arguments:

```cpp
auto model = createLlamaCppModel("/path/to/model.gguf");
Agent agent(model.value());
agent.setSystemPrompt("You are a helpful assistant.")
     .setTemperature(0.7F)
     .registerTool(createTool(
         "get_labels", "Return available track labels.",
         {{"format", "The desired output format", "string", false}},
         [](const JsonObject&) -> ToolResult {
           return {true, "[\\"drums\\",\\"bass\\",\\"vocals\\"]", {}};
         }));
auto conversation = agent.createConversation();
auto result = conversation.run("Say hello.");
```

Keep the conceptual model explicit:

- `Model` is the inference engine.
- `Agent` owns reusable behavior, prompt, tools, and options.
- `Conversation` owns one independent transcript and run state.
- `Tool` combines a name, description, parameter list, and callback.

The typed core (`AgentSpec`, `GenerationRequest`, `ToolDefinition`,
`Expected<T>`, and related types) remains available for advanced users and
internal correctness, but should not make the normal path feel ceremonial.
When adding capabilities, expose the simple facade first and retain the typed
escape hatch. Preserve compatibility aliases such as
`start_conversation()` when introducing friendlier names such as
`createConversation()`.

## Tests

- Add or update Catch2 tests for behavior changes.
- Prefer `FakeModel` scripts for deterministic agent-loop tests; tests must not
  require a GGUF model or network inference.
- Cover both the successful path and relevant failures, including history and
  callback ordering when behavior is externally observable.
- Run the narrowest relevant build while iterating, then run `./build.sh` before
  handing off changes when dependencies are available.
- If full validation cannot run because dependency fetching or a local model is
  unavailable, report exactly what was and was not verified.

## Change discipline

- Read the relevant public headers, implementation, and tests before changing
  behavior.
- Preserve existing user changes in the working tree and avoid modifying
  unrelated files.
- Do not commit generated build output, downloaded dependencies, or GGUF model
  files.
- Treat `README.md` as the primary user guide. It should provide a clear path
  through project requirements, onboarding, building, usage, and contributing.
- Update `README.md` whenever a change affects that user journey, including
  public API usage, prerequisites, build options, examples, contribution
  workflows, or known limitations.
