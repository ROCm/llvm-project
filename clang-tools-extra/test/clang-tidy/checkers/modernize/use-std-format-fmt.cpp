// RUN: %check_clang_tidy %s modernize-use-std-format %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:              modernize-use-std-format.StrictMode: true, \
// RUN:              modernize-use-std-format.StrFormatLikeFunctions: 'fmt::sprintf', \
// RUN:              modernize-use-std-format.ReplacementFormatFunction: 'fmt::format', \
// RUN:              modernize-use-std-format.FormatHeader: '<fmt/core.h>' \
// RUN:            }}" \
// RUN:   -- -isystem %clang_tidy_headers

// CHECK-FIXES: #include <fmt/core.h>
#include <string>

namespace fmt
{
template <typename S, typename... T,
          typename Char = char>
std::basic_string<Char> sprintf(const S& fmt, const T&... args);
} // namespace fmt

std::string fmt_sprintf_simple() {
  return fmt::sprintf("Hello %s %d", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'fmt::format' instead of 'sprintf' [modernize-use-std-format]
  // CHECK-FIXES: return fmt::format("Hello {} {}", "world", 42);
}
