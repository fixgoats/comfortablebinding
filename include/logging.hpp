#pragma once
#include <iostream>

enum LogLevel { None = 0, Debug = 1 };

struct Logger {
  LogLevel level;
  Logger() : level{LogLevel::None} {};
  Logger(Logger& other) = delete;
  void operator=(const Logger&) = delete;

  static Logger* get() { return logger_; }
  template <class Arg, class... Args>
  void debug(std::ostream& out, Arg&& arg, Args&&... args) {
    if (level >= LogLevel::Debug) {
      out << std::forward<Arg>(arg);
      ((out << ' ' << std::forward<Args>(args)), ...);
      out << std::endl;
    }
  }

private:
  static Logger* logger_;
};

static Logger logger{};
