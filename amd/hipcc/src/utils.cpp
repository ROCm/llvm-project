#include "utils.h"
#include "filesystem.h"

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#include <tchar.h>
#include <windows.h>
#ifdef _UNICODE
typedef wchar_t TCHAR;
typedef std::wstring TSTR;
typedef std::wstring::size_type TSIZE;
#define ENDLINE L"/\\"
#else
typedef char TCHAR;
typedef std::string TSTR;
typedef std::string::size_type TSIZE;
#define ENDLINE "/\\"
#endif
#else
#if defined(__APPLE__)
#include <limits.h>
#include <mach-o/dyld.h>
#include <cstdlib>
#endif
#include <unistd.h>
#endif

#include <iostream>
#include <sstream>

std::string hipcc::utils::getSelfPath() {
  constexpr size_t MAX_PATH_CHAR = 1024;
  std::string path;
#if defined(_WIN32) || defined(_WIN64)
  TCHAR buffer[MAX_PATH] = {0};
  GetModuleFileName(NULL, buffer, MAX_PATH_CHAR);
  TSIZE pos = TSTR(buffer).find_last_of(ENDLINE);
  TSTR wide = TSTR(buffer).substr(0, pos);
  path = std::string(wide.begin(), wide.end());
#elif defined(__APPLE__)
  char buff[MAX_PATH_CHAR];
  uint32_t size = sizeof(buff);
  std::string exePathString;
  if (_NSGetExecutablePath(buff, &size) == 0) {
    exePathString = buff;
  } else {
    std::vector<char> dynamicBuff(size);
    if (_NSGetExecutablePath(dynamicBuff.data(), &size) == 0) {
      exePathString = dynamicBuff.data();
    }
  }
  if (!exePathString.empty()) {
    char resolved[PATH_MAX];
    const char *resolvedPath = realpath(exePathString.c_str(), resolved);
    fs::path exePath(resolvedPath != nullptr ? resolvedPath : exePathString);
    path = exePath.parent_path().string();
  } else {
    std::cerr << "_NSGetExecutablePath: Error reading the exe path" << std::endl;
    exit(-1);
  }
#else
  char buff[MAX_PATH_CHAR];
  ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff) - 1);
  if (len > 0) {
    buff[len] = '\0';
    path = std::string(buff);
    fs::path exePath(path);
    path = exePath.parent_path().string();
  } else {
    std::cerr << "readlink: Error reading the exe path" << std::endl;
    perror("readlink");
    exit(-1);
  }
#endif
  return path;
}

std::vector<std::string> hipcc::utils::splitStr(std::string const &fullStr,
                                                char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream check1(fullStr);
  std::string intermediate;
  while (std::getline(check1, intermediate, delimiter)) {
    tokens.emplace_back(std::move(intermediate));
  }
  return tokens;
}
