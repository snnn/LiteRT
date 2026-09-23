/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Synthetic FIFO tests only: no model loading, inference, or performance test.
#include "tensor/examples/gemma4/native/decode_trace.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {
using litert::tensor::examples::gemma4::native::DecodeTrace;
using litert::tensor::examples::gemma4::native::DecodeTraceSection;

int checks = 0;
#define CHECK(condition)                                                \
  do {                                                                  \
    ++checks;                                                           \
    if (!(condition)) {                                                 \
      std::cerr << __FILE__ << ':' << __LINE__ << ": " #condition "\n"; \
      std::exit(1);                                                     \
    }                                                                   \
  } while (false)

class Fixture {
 public:
  Fixture() {
    const char* tmp = std::getenv("TEST_TMPDIR");
    if (!tmp) tmp = std::getenv("TMPDIR");
#ifdef __ANDROID__
    if (!tmp) tmp = "/data/local/tmp";
#else
    if (!tmp) tmp = "/tmp";
#endif
    std::string pattern = std::string(tmp) + "/native-decode-trace-XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    CHECK(mkdtemp(writable.data()) != nullptr);
    dir_ = writable.data();
    path_ = dir_ + "/markers";
    const int fifo_result = mkfifo(path_.c_str(), 0600);
    if (fifo_result != 0) {
      const int error = errno;
      std::cerr << "mkfifo(" << path_ << "): " << std::strerror(error)
                << " (errno " << error << ")\n";
    }
    CHECK(fifo_result == 0);
  }
  ~Fixture() {
    if (reader_ >= 0) close(reader_);
    unlink(path_.c_str());
    rmdir(dir_.c_str());
    unsetenv("LITERT_NATIVE_TRACE_FIFO");
  }
  void Configure() {
    CHECK(setenv("LITERT_NATIVE_TRACE_FIFO", path_.c_str(), 1) == 0);
  }
  void OpenReader() {
    reader_ = open(path_.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    CHECK(reader_ >= 0);
  }
  void CloseReader() {
    CHECK(close(reader_) == 0);
    reader_ = -1;
  }
  std::string Read() {
    std::string result;
    std::array<char, 512> bytes;
    while (true) {
      const ssize_t count = read(reader_, bytes.data(), bytes.size());
      if (count > 0)
        result.append(bytes.data(), count);
      else {
        CHECK(count == 0 || errno == EAGAIN || errno == EWOULDBLOCK);
        return result;
      }
    }
  }
  const std::string& path() const { return path_; }

 private:
  std::string dir_, path_;
  int reader_ = -1;
};

void Disabled() {
  CHECK(unsetenv("LITERT_NATIVE_TRACE_FIFO") == 0);
  DecodeTrace trace;
  CHECK(trace.InitializeFromEnvironment().ok());
  CHECK(!trace.enabled());
  DecodeTraceSection section(trace, true);
  CHECK(section.Begin().ok());
  CHECK(section.End().ok());
  CHECK(section.elapsed_ms() == 0);
}

void ProtocolAndWarmups() {
  Fixture fifo;
  fifo.Configure();
  fifo.OpenReader();
  DecodeTrace trace;
  CHECK(trace.InitializeFromEnvironment().ok());
  CHECK(trace.enabled());
  CHECK(!trace.InitializeFromEnvironment().ok());
  {
    DecodeTraceSection warmup(trace, false);
    CHECK(warmup.Begin().ok());
    CHECK(warmup.End().ok());
    CHECK(warmup.elapsed_ms() == 0);
  }
  CHECK(fifo.Read().empty());
  for (int repetition = 0; repetition < 2; ++repetition) {
    DecodeTraceSection section(trace, true);
    CHECK(!section.End().ok());
    CHECK(section.Begin().ok());
    CHECK(!section.Begin().ok());
    CHECK(section.End().ok());
    CHECK(section.elapsed_ms() >= 0);
    CHECK(!section.End().ok());
    CHECK(!section.Begin().ok());
  }
  CHECK(fifo.Read() ==
        "section_begin decode\nsection_end decode\n"
        "section_begin decode\nsection_end decode\n");
  {
    DecodeTraceSection early_return(trace, true);
    CHECK(early_return.Begin().ok());
  }
  CHECK(fifo.Read() == "section_begin decode\nsection_end decode\n");
}

void InvalidChannels() {
  CHECK(setenv("LITERT_NATIVE_TRACE_FIFO", "", 1) == 0);
  DecodeTrace empty;
  CHECK(!empty.InitializeFromEnvironment().ok());
  Fixture fifo;
  fifo.Configure();
  DecodeTrace no_reader;
  CHECK(!no_reader.InitializeFromEnvironment().ok());
  CHECK(!no_reader.enabled());
  CHECK(unlink(fifo.path().c_str()) == 0);
  DecodeTrace missing;
  CHECK(!missing.InitializeFromEnvironment().ok());
  int fd = open(fifo.path().c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  CHECK(fd >= 0);
  CHECK(write(fd, "untouched", 9) == 9);
  CHECK(close(fd) == 0);
  DecodeTrace regular;
  CHECK(!regular.InitializeFromEnvironment().ok());
  CHECK(!regular.enabled());
  fd = open(fifo.path().c_str(), O_RDONLY);
  CHECK(fd >= 0);
  char content[10] = {};
  CHECK(read(fd, content, sizeof(content)) == 9);
  CHECK(std::string(content) == "untouched");
  CHECK(close(fd) == 0);
}

void DisconnectedReader() {
  Fixture fifo;
  fifo.Configure();
  fifo.OpenReader();
  DecodeTrace trace;
  CHECK(trace.InitializeFromEnvironment().ok());
  DecodeTraceSection section(trace, true);
  CHECK(section.Begin().ok());
  CHECK(fifo.Read() == "section_begin decode\n");
  fifo.CloseReader();
  // The default SIGPIPE disposition would kill this process without the
  // writer's thread-local signal handling.
  CHECK(!section.End().ok());
  CHECK(section.elapsed_ms() == 0);
  sigset_t pending;
  CHECK(sigpending(&pending) == 0);
  CHECK(sigismember(&pending, SIGPIPE) == 0);

  sigset_t pipe_signal, old_mask;
  sigemptyset(&pipe_signal);
  sigaddset(&pipe_signal, SIGPIPE);
  CHECK(pthread_sigmask(SIG_BLOCK, &pipe_signal, &old_mask) == 0);
  CHECK(pthread_kill(pthread_self(), SIGPIPE) == 0);
  CHECK(!trace.Write(true).ok());
  CHECK(sigpending(&pending) == 0);
  CHECK(sigismember(&pending, SIGPIPE) == 1);
  struct timespec no_wait{};
  CHECK(sigtimedwait(&pipe_signal, nullptr, &no_wait) == SIGPIPE);
  CHECK(pthread_sigmask(SIG_SETMASK, &old_mask, nullptr) == 0);
}

void FullPipe() {
  Fixture fifo;
  fifo.Configure();
  fifo.OpenReader();
  DecodeTrace trace;
  CHECK(trace.InitializeFromEnvironment().ok());
  int filler = open(fifo.path().c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
  CHECK(filler >= 0);
  std::array<char, 4096> zeros{};
  while (write(filler, zeros.data(), zeros.size()) > 0) {
  }
  CHECK(errno == EAGAIN || errno == EWOULDBLOCK);
  CHECK(close(filler) == 0);
  DecodeTraceSection section(trace, true);
  CHECK(!section.Begin().ok());
  CHECK(section.elapsed_ms() == 0);
}
}  // namespace

int main() {
  Disabled();
  ProtocolAndWarmups();
  InvalidChannels();
  DisconnectedReader();
  FullPipe();
  std::cout << "Passed " << checks << " decode trace checks\n";
  return 0;
}
