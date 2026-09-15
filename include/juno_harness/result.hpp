/**
 * @file result.hpp
 * @brief Defines the Result class and related types for error handling in the Juno inference model.
 */

#pragma once

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace juno::harness {

/** Error codes for the Juno inference model. */
enum class ErrorCode {
  InvalidConfiguration,
  ModelUnavailable,
  ModelLoadFailed,
  GenerationFailed,
  InvalidModelOutput,
  ToolExecutionFailed,
  ContextLimitExceeded,
  IterationLimitExceeded,
  Cancelled,
};

/** Represents an error in the Juno inference model. */
struct Error {
  ErrorCode code;
  std::string message;
};

/** Represents the result of an operation in the Juno inference model. */
template <typename T>
class Result {
 public:
  Result(T value) : value_(std::move(value)) {}
  template <typename U>
    requires(!std::same_as<std::remove_cvref_t<U>, Result> && std::constructible_from<T, U&&>)
  Result(U&& value) : value_(T(std::forward<U>(value))) {}
  Result(Error error) : value_(std::move(error)) {}

  [[nodiscard]] bool has_value() const { return std::holds_alternative<T>(value_); }
  [[nodiscard]] explicit operator bool() const { return has_value(); }
  [[nodiscard]] T& value() { return std::get<T>(value_); }
  [[nodiscard]] const T& value() const { return std::get<T>(value_); }
  [[nodiscard]] Error& error() { return std::get<Error>(value_); }
  [[nodiscard]] const Error& error() const { return std::get<Error>(value_); }

 private:
  std::variant<T, Error> value_;
};

/** Specialization of Result for void type. */
template <>
class Result<void> {
 public:
  Result() : error_(std::monostate{}) {}
  Result(Error error) : error_(std::move(error)) {}

  [[nodiscard]] bool has_value() const { return std::holds_alternative<std::monostate>(error_); }
  [[nodiscard]] explicit operator bool() const { return has_value(); }
  [[nodiscard]] Error& error() { return std::get<Error>(error_); }
  [[nodiscard]] const Error& error() const { return std::get<Error>(error_); }

 private:
  std::variant<std::monostate, Error> error_;
};

}  // namespace juno::harness
