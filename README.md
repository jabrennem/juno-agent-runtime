# juno-harness-cpp
A native modern C++ agent harness for real-time creative and production workflows. Initially powered by llama.cpp.

Juno Harness SDK is a C++20 SDK for embedding a small, tool-using agent loop in native applications. It separates agent orchestration, tool execution, in-memory conversation state, and inference models. The deterministic fake model is available as build-only test support; llama.cpp is enabled by default for local GGUF inference.

## Prerequisites

- CMake 3.20 or newer.
- A C++20 compiler: Apple Clang on macOS, or Clang/GCC on Linux.
- Git and network access when CMake is fetching dependencies.
- A GGUF model only when running the llama.cpp playgrounds or a real-model smoke test.

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

The llama.cpp model can also be enabled explicitly:

```sh
JUNO_HARNESS_BUILD_DIR=build-llama ./build.sh -DJUNO_HARNESS_ENABLE_LLAMA_CPP=ON
```

Useful CMake options:

| Option | Default | Purpose |
| --- | --- | --- |
| `JUNO_HARNESS_BUILD_TESTS` | `OFF` | Build the Catch2 unit test executable and register it with CTest. |
| `JUNO_HARNESS_BUILD_EXAMPLES` | `ON` | Build the three music-workflow playgrounds. |
| `JUNO_HARNESS_ENABLE_LLAMA_CPP` | `ON` | Fetch and compile the in-process llama.cpp model. Set to `OFF` for a model-free build. |
| `JUNO_HARNESS_FETCH_DEPS` | `ON` | Fetch pinned dependencies; set `OFF` to use installed packages. |

`build.sh` accepts any additional CMake cache arguments. Set `JUNO_HARNESS_BUILD_TYPE=Release` for an optimized build, set `JUNO_HARNESS_BUILD_DIR` to choose the build directory, and set `JUNO_HARNESS_SKIP_TESTS=1` when you only want compilation. If CMake is installed outside your `PATH`, set `CMAKE_BIN=/path/to/cmake` (and `CTEST_BIN=/path/to/ctest`).

## Run the playgrounds

Each playground takes a path to a local GGUF model:

```sh
./build-llama/juno_session_prep_planning_playground /absolute/path/to/model.gguf
./build-llama/juno_session_prep_playground /absolute/path/to/model.gguf
./build-llama/juno_mix_playground /absolute/path/to/model.gguf
```

The planning playground creates a session-preparation plan from raw WAV metadata and a mix template. The session-prep and mix playgrounds are independent ad-hoc agent loops with tools restricted to their respective domains. Each supports `/help`, `/clear`, `/history`, and `/quit`. The example tools simulate DAW operations and are intended to be replaced with calls into a production project service. Tool-capable models need a compatible chat template. Juno Harness uses the model’s template by default; `LlamaCppConfig::chat_template_override` can supply a known compatible template name.

## Unit tests

```sh
cmake -S . -B build-test -DJUNO_HARNESS_BUILD_TESTS=ON
cmake --build build-test
ctest --test-dir build-test --output-on-failure
```

The automated tests link the build-only `juno_harness_test_support` target, which provides `FakeModel`; they do not download a model. They cover final responses, tool loops, unknown tools, iteration limits, callbacks, handler-owned argument validation, and independent conversation histories.

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
#include <string_view>
#include <utility>
#include "juno_harness/juno_harness.hpp"

auto model_result = juno::harness::LlamaCppModel::create({.model_path = "/path/to/model.gguf"});
if (!model_result) return 1;
auto model = model_result.value();
juno::harness::Agent agent(model, {.system_prompt = "Be concise."});
auto conversation = agent.start_conversation();
auto result = conversation.run("Say hello.", [](const juno::harness::AgentEvent& event) {
  // TextDelta, ToolStarted, ToolCompleted, Completed, or Error.
});
```

Tools bundle a `ToolDefinition` with a handler returning `Result<std::string>`. The handler receives the model's argument JSON and owns argument validation and result serialization:

```cpp
auto spec = juno::harness::AgentSpec{
    .system_prompt = "Be concise.",
    .tools = {{{
        {"get_labels", "Return the available track labels.", R"({"type":"object"})"},
        [](std::string_view) -> juno::harness::Result<std::string> {
          return R"(["drums","bass","guitars","vocals"])");
        }}}};
juno::harness::Agent agent(model, std::move(spec));
auto conversation = agent.start_conversation();
```

Tool schemas, call arguments, and results use JSON strings. The same agent definition can create many independent conversations; each conversation owns its transcript and can be cleared without affecting the others.

## Runtime model and limitations

- `Agent` holds a reusable model and immutable shared `AgentSpec`; every `Conversation` owns an in-memory transcript.
- Conversations are not reentrant or thread-safe. Use one conversation per concurrent chat.
- Calls and event callbacks run synchronously, in model order.
- Tool errors—including unknown tools and handler exceptions—are appended as tool-result messages so a model can recover on its next turn.
- The v1 loop explicitly fails when its context or inference-turn limit is exceeded. It does not yet summarize, truncate, persist, or retrieve memory.
- `LlamaCppModel` is in-process and uses RAII to manage the model and per-run inference contexts. Its llama.cpp dependency is isolated from the SDK’s public headers.
- API-key/cloud models are not implemented yet; implement `Model` to add one without changing `Agent` or `Conversation`.

## Layout

| Path | Responsibility |
| --- | --- |
| `include/juno_harness` | Public SDK API: runtime, messages, tools, errors, and production models. |
| `src/agent.cpp` | Agent loop, conversation transcript, tool dispatch, and events. |
| `test_support/fake_model.cpp` | Deterministic scripted model for build-only tests. |
| `test_support/juno_harness/fake_model.hpp` | Build-only fake model API for tests. |
| `src/llama_cpp_model.cpp` | Private direct llama.cpp adapter. |
| `examples` | Independent planning, session-prep, and mixing playground executables. |
| `tests` | Fake-model unit tests. |
