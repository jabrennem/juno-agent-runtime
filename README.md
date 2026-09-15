# juno-harness-cpp
A native modern C++ agent harness for real-time creative and production workflows. Initially powered by llama.cpp.

Juno Harness SDK is a C++20 SDK for embedding a small, tool-using agent loop in native applications. It separates agent orchestration, tool execution, in-memory conversation state, and inference backends. The deterministic fake backend is available as build-only test support; llama.cpp is enabled by default for local GGUF inference.

## Prerequisites

- CMake 3.20 or newer.
- A C++20 compiler: Apple Clang on macOS, or Clang/GCC on Linux.
- Git and network access when CMake is fetching dependencies.
- A GGUF model only when running the llama.cpp playground or real-model smoke test.

On macOS, llama.cpp selects Metal support when it is available. Linux defaults to CPU; configure llama.cpp’s own CMake options in a parent build if you need CUDA, Vulkan, or another accelerator.

## Build

The simplest path is the repository build helper. It configures CMake, builds the library and playgrounds, and runs CTest:

```sh
./build.sh
```

The build does not need a model, but it does fetch and compile llama.cpp by default:

```sh
cmake -S . -B build -DJUNO_HARNESS_BUILD_EXAMPLES=ON
cmake --build build
```

The llama backend can also be enabled explicitly:

```sh
JUNO_HARNESS_BUILD_DIR=build-llama ./build.sh -DJUNO_HARNESS_ENABLE_LLAMA_CPP=ON
```

Useful CMake options:

| Option | Default | Purpose |
| --- | --- | --- |
| `JUNO_HARNESS_BUILD_TESTS` | `OFF` | Build the Catch2 unit test executable and register it with CTest. |
| `JUNO_HARNESS_BUILD_EXAMPLES` | `ON` | Build the llama playground. |
| `JUNO_HARNESS_ENABLE_LLAMA_CPP` | `ON` | Fetch and compile the in-process llama.cpp backend. Set to `OFF` for a backend-free build. |
| `JUNO_HARNESS_FETCH_DEPS` | `ON` | Fetch pinned dependencies; set `OFF` to use installed packages. |

`build.sh` accepts any additional CMake cache arguments. Set `JUNO_HARNESS_BUILD_TYPE=Release` for an optimized build, set `JUNO_HARNESS_BUILD_DIR` to choose the build directory, and set `JUNO_HARNESS_SKIP_TESTS=1` when you only want compilation. If CMake is installed outside your `PATH`, set `CMAKE_BIN=/path/to/cmake` (and `CTEST_BIN=/path/to/ctest`).

## Run the playground

The llama playground takes a path to a local GGUF model and opens an interactive chat:

```sh
./build-llama/juno_harness_playground /absolute/path/to/model.gguf
```

Enter prompts at the `>` prompt. Use `/help`, `/clear`, `/history`, or `/quit` to control the session. Edit the system prompt and context settings in [examples/llama_playground.cpp](/Users/joshbrenneman/Dev/juno-harness-cpp/examples/llama_playground.cpp). Tool-capable models need a compatible chat template. Juno Harness uses the model’s template by default; `LlamaCppConfig::chat_template_override` can supply a known compatible template name.

## Unit tests

```sh
cmake -S . -B build-test -DJUNO_HARNESS_BUILD_TESTS=ON
cmake --build build-test
ctest --test-dir build-test --output-on-failure
```

The automated tests link the build-only `juno_harness_test_support` target, which provides `FakeBackend`; they do not download a model. They cover final responses, tool loops, malformed calls, unknown tools, iteration limits, callbacks, and independent session histories.

## Use from another CMake application

Install Juno Harness SDK first:

```sh
cmake --install build --prefix /desired/prefix
```

Then consume its exported target:

```cmake
find_package(JunoHarness CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Juno::Harness)
```

For a parent-project build, add the checkout through `FetchContent` instead:

```cmake
include(FetchContent)
FetchContent_Declare(juno_harness SOURCE_DIR /absolute/path/to/juno-harness-cpp)
FetchContent_MakeAvailable(juno_harness)
target_link_libraries(my_app PRIVATE Juno::Harness)
```

Minimal SDK use:

```cpp
#include <memory>
#include "juno_harness/juno_harness.hpp"

auto backend_result = juno::harness::LlamaCppBackend::create({.model_path = "/path/to/model.gguf"});
if (!backend_result) return 1;
auto backend = backend_result.value();
juno::harness::Agent agent(backend, {.system_prompt = "Be concise."});
auto session = agent.create_session();
auto result = session.run("Say hello.", [](const juno::harness::AgentEvent& event) {
  // TextDelta, ToolStarted, ToolCompleted, Completed, or Error.
});
```

Register a tool with a `ToolDefinition` and a handler returning `Result<std::string>`. Tool schemas, call arguments, and results are JSON strings. Juno validates that a call is JSON-shaped; application handlers own detailed schema validation.

## Runtime model and limitations

- `Agent` holds reusable configuration and registered tools; every `AgentSession` owns an in-memory transcript.
- Sessions are not reentrant or thread-safe. Use one session per concurrent conversation.
- Calls and event callbacks run synchronously, in model order.
- Tool errors—including unknown tools and handler exceptions—are appended as tool-result messages so a model can recover on its next turn.
- The v1 loop explicitly fails when its context or inference-turn limit is exceeded. It does not yet summarize, truncate, persist, or retrieve memory.
- `LlamaCppBackend` is in-process and uses RAII to manage the model and per-run inference contexts. Its llama.cpp dependency is isolated from the SDK’s public headers.
- API-key/cloud backends are not implemented yet; implement `InferenceBackend` to add one without changing `Agent` or `AgentSession`.

## Layout

| Path | Responsibility |
| --- | --- |
| `include/juno_harness` | Public SDK API: runtime, messages, tools, errors, and production backends. |
| `src/agent.cpp` | Agent loop, session transcript, tool dispatch, and events. |
| `test_support/fake_backend.cpp` | Deterministic scripted backend for build-only tests. |
| `test_support/juno_harness/fake_backend.hpp` | Build-only fake backend API for tests. |
| `src/llama_cpp_backend.cpp` | Private direct llama.cpp adapter. |
| `examples` | Editable GGUF playground executable. |
| `tests` | Fake-backend unit tests. |
