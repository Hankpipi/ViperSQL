/* Copyright (c) 2026, Zihao Yu.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License, version 2.0,
  as published by the Free Software Foundation.
*/

#include "sql/semantic_operator_runtime_process.h"

#ifndef _WIN32

#include <errno.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/prctl.h>
#endif

#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

namespace semantic_operator_runtime {
namespace {

std::mutex process_mutex;
pid_t runtime_pid{-1};

bool directory_is_accessible(const std::string &path) {
  return !path.empty() && access(path.c_str(), R_OK | X_OK) == 0;
}

bool executable_is_accessible(const std::string &path) {
  return !path.empty() && access(path.c_str(), X_OK) == 0;
}

void reap_blocking(pid_t pid) {
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
}

bool reap_if_exited(pid_t pid) {
  int status = 0;
  pid_t result;
  do {
    result = waitpid(pid, &status, WNOHANG);
  } while (result < 0 && errno == EINTR);
  return result == pid || (result < 0 && errno == ECHILD);
}

void terminate_and_reap(pid_t pid) {
  if (pid <= 0) return;
  if (kill(-pid, SIGTERM) != 0 && errno != ESRCH) kill(pid, SIGTERM);
  for (int attempt = 0; attempt < 100; ++attempt) {
    if (reap_if_exited(pid)) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (kill(-pid, SIGKILL) != 0 && errno != ESRCH) kill(pid, SIGKILL);
  reap_blocking(pid);
}

bool health_check(const std::string &endpoint) {
  try {
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::req);
    socket.set(zmq::sockopt::linger, 0);
    socket.set(zmq::sockopt::sndtimeo, 250);
    socket.set(zmq::sockopt::rcvtimeo, 250);
    socket.connect(endpoint);
    const nlohmann::json request = {
        {"name", "runtime_health"}, {"values", nlohmann::json::array()}};
    const std::string request_bytes = request.dump();
    if (!socket.send(zmq::buffer(request_bytes), zmq::send_flags::none))
      return false;
    zmq::message_t reply;
    if (!socket.recv(reply, zmq::recv_flags::none)) return false;
    const auto response = nlohmann::json::parse(
        static_cast<const char *>(reply.data()),
        static_cast<const char *>(reply.data()) + reply.size());
    return response.value("ok", false) &&
           response.value("name", std::string()) == "runtime_health" &&
           response.value("status", std::string()) == "ready" &&
           response.value("protocol_version", 0) == 1 &&
           response.value("server_epoch", std::string()).size() == 32;
  } catch (...) {
    return false;
  }
}

pid_t spawn(const Process_options &options, std::string *error) {
  if (!directory_is_accessible(options.runtime_root) ||
      access((options.runtime_root + "/__main__.py").c_str(), R_OK) != 0) {
    *error = "semantic runtime package directory is not accessible: " +
             options.runtime_root;
    return -1;
  }
  if (!executable_is_accessible(options.python_executable)) {
    *error = "semantic runtime Python executable is not accessible: " +
             options.python_executable;
    return -1;
  }

  long open_max = sysconf(_SC_OPEN_MAX);
  if (open_max < 0 || open_max > 65536) open_max = 65536;
  const std::string::size_type separator = options.runtime_root.find_last_of('/');
  if (separator == std::string::npos || separator == 0) {
    *error = "semantic runtime root must be an absolute package path";
    return -1;
  }
  const std::string package_parent = options.runtime_root.substr(0, separator);
  const pid_t parent_pid = getpid();
  const pid_t pid = fork();
  if (pid < 0) {
    *error = std::string("could not fork semantic runtime: ") +
             std::strerror(errno);
    return -1;
  }
  if (pid == 0) {
    setpgid(0, 0);
#ifdef __linux__
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || getppid() != parent_pid)
      _exit(125);
#else
    if (getppid() != parent_pid) _exit(125);
#endif
    if (chdir(package_parent.c_str()) != 0) _exit(126);
    for (int descriptor = 3; descriptor < open_max; ++descriptor)
      close(descriptor);
    execl(options.python_executable.c_str(),
          options.python_executable.c_str(), "-m", "semantic_operator_runtime",
          "serve", "--endpoint", options.endpoint.c_str(),
          static_cast<char *>(nullptr));
    _exit(127);
  }
  if (setpgid(pid, pid) != 0 && errno != EACCES && errno != ESRCH) {
    terminate_and_reap(pid);
    *error = std::string("could not isolate semantic runtime process group: ") +
             std::strerror(errno);
    return -1;
  }
  return pid;
}

}  // namespace

bool start(const Process_options &options, std::string *error) {
  std::lock_guard<std::mutex> guard(process_mutex);
  if (error == nullptr) return false;
  error->clear();
  if (runtime_pid > 0) {
    *error = "semantic runtime is already owned by this mysqld process";
    return false;
  }
  if (options.endpoint.empty()) {
    *error = "semantic runtime endpoint must not be empty";
    return false;
  }
  if (health_check(options.endpoint)) {
    *error = "semantic runtime endpoint is already occupied";
    return false;
  }

  const pid_t child = spawn(options, error);
  if (child <= 0) return false;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(options.startup_timeout_seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (reap_if_exited(child)) {
      *error = "semantic runtime exited before becoming ready";
      return false;
    }
    if (health_check(options.endpoint)) {
      runtime_pid = child;
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  terminate_and_reap(child);
  *error = "semantic runtime did not become ready before the startup timeout";
  return false;
}

void stop() {
  std::lock_guard<std::mutex> guard(process_mutex);
  if (runtime_pid <= 0) return;
  const pid_t child = runtime_pid;
  runtime_pid = -1;
  terminate_and_reap(child);
}

bool is_running() {
  std::lock_guard<std::mutex> guard(process_mutex);
  if (runtime_pid <= 0) return false;
  if (reap_if_exited(runtime_pid)) {
    runtime_pid = -1;
    return false;
  }
  return true;
}

}  // namespace semantic_operator_runtime

#else

namespace semantic_operator_runtime {

bool start(const Process_options &, std::string *error) {
  if (error != nullptr)
    *error = "automatic semantic runtime lifecycle is unavailable on Windows";
  return false;
}

void stop() {}

bool is_running() { return false; }

}  // namespace semantic_operator_runtime

#endif
