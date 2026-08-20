/* Copyright (c) 2026, Zihao Yu.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License, version 2.0,
  as published by the Free Software Foundation.
*/

#ifndef SQL_SEMANTIC_OPERATOR_RUNTIME_PROCESS_H_
#define SQL_SEMANTIC_OPERATOR_RUNTIME_PROCESS_H_

#include <string>

namespace semantic_operator_runtime {

/** Configuration for the semantic runtime child owned by mysqld. */
struct Process_options {
  std::string python_executable;
  std::string runtime_root;
  std::string endpoint;
  unsigned int startup_timeout_seconds{15};
};

/**
  Start the runtime and wait for its authenticated health response.

  @returns true on success; false with a human-readable error otherwise.
*/
bool start(const Process_options &options, std::string *error);

/** Terminate and reap the runtime child. Safe to call more than once. */
void stop();

/** Return true while an owned child process is present. */
bool is_running();

}  // namespace semantic_operator_runtime

#endif  // SQL_SEMANTIC_OPERATOR_RUNTIME_PROCESS_H_
