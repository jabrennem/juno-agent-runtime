cmake_minimum_required(VERSION 3.20)
project(JunoAgentRuntimePackageConsumer LANGUAGES CXX)

find_package(JunoAgentRuntime CONFIG REQUIRED)
add_executable(juno_package_consumer main.cpp)
target_link_libraries(juno_package_consumer PRIVATE Juno::Agent)
