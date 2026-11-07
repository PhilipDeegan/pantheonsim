// Launch-level result caching ("memoisation"), opt-in: VGPU_MEMO=1. docs/perf-design/03-result-memoisation.md is the
// study; docs/performance.md "Round sp35" says what was built and measured. This header holds the parts the memory
// manager needs to know about: the hash, the thread scope that tells a host call from a kernel's, and the event sink
// the manager reports host-side changes to. The cache itself is src/runtime/memo.cpp.
//
// The idea in one paragraph. A device's memory state is named by its HISTORY, not its content: S(n+1) = H(S(n), what
// happened next). A host write, fill, copy, allocation or free is an event hashed into S (a write with the bytes it
// writes, which are already O(bytes) to copy). A launch is an event whose description (kernel code, grid, block,
// arguments, environment, and the S before it) is its cache key. A launch the cache cannot represent POISONS the
// state: S becomes a fresh unique value, so nothing before it can match anything after it. A launch that changes no
// byte leaves S as it was, which is what lets a loop of idempotent kernels hit within one run.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace vgpu::memo {

struct Hash128 {
  uint64_t a = 0, b = 0;
  bool operator==(const Hash128& o) const { return a == o.a && b == o.b; }
  bool operator!=(const Hash128& o) const { return !(*this == o); }
};

// A hash of `n` bytes (32 bytes per step in four lanes; the same hash on every host).
Hash128 hash_bytes(const void* p, size_t n);
// The hash of one page (a 64-bit fold of the above's lanes, with the length mixed in).
uint64_t page_hash(const uint8_t* p, size_t n);

inline uint64_t mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}

// A streaming hash of words and byte strings, 128 bits out. Not cryptographic: it guards against accident, not attack
// (an entry on disk is as trusted as the program that wrote it).
class Hasher {
 public:
  explicit Hasher(uint64_t tag = 0) { u64(tag); }
  void u64(uint64_t v) {
    a_ = (a_ ^ v) * 0x9e3779b97f4a7c15ull;
    a_ ^= a_ >> 29;
    b_ = (b_ + v + 0x632be59bd9b4e019ull) * 0xc2b2ae3d27d4eb4full;
    b_ = (b_ << 23) | (b_ >> 41);
    a_ += b_ >> 7;
  }
  void hash(const Hash128& h) { u64(h.a), u64(h.b); }
  // The length, then the bytes: two different splits of the same bytes are different inputs.
  void bytes(const void* p, size_t n) {
    u64(n);
    hash(hash_bytes(p, n));
  }
  void str(const char* s);
  Hash128 done() const { return Hash128{mix64(a_ ^ (b_ * 0x9e3779b97f4a7c15ull)), mix64(b_ ^ (a_ * 0xff51afd7ed558ccdull) ^ 0x5bd1e995ull)}; }

 private:
  uint64_t a_ = 0x243f6a8885a308d3ull, b_ = 0x13198a2e03707344ull;
};

// Set by an engine when a kernel reads %smid in the one case where the value depends on which host thread ran the block
// (a grid with more blocks than SMs): what that kernel computes then is not a function of its inputs, so a launch that
// sets it on more than one thread is not cached.
inline std::atomic<bool> g_smid_by_worker{false};

// Who is calling into the memory manager. A host call (the runtime's cudaMemcpy, a library's write) is an event in the
// device's history. A kernel's own writes are not: the launch's before/after diff holds them. The engine's worker
// threads set kEngine for good; the thread inside Device::launch sets it for the length of the launch; replaying a hit
// sets kReplay so the replay's own writes are not events either.
enum Scope : int { kHost = 0, kEngine = 1, kReplay = 2 };
inline thread_local int t_scope = kHost;
class ScopeGuard {
 public:
  explicit ScopeGuard(int s) : prev_(t_scope) { t_scope = s; }
  ~ScopeGuard() { t_scope = prev_; }
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;

 private:
  int prev_;
};

// What the memory manager reports. Every call comes from inside the manager, before the change is made (a call that
// then fails is still an event: the history is of calls, and the same calls fail the same way). Which scope the call
// came from is read from t_scope by the sink.
struct EventSink {
  virtual ~EventSink() = default;
  virtual void on_write(uint64_t dst, const void* src, uint64_t len) = 0;
  virtual void on_fill(uint64_t dst, const uint8_t* pattern, uint32_t pattern_len, uint64_t len) = 0;
  // A device-to-device copy the manager did without a staging buffer: the bytes are a function of the state, so the
  // addresses are the whole event.
  virtual void on_copy(uint64_t dst, uint64_t src, uint64_t len) = 0;
  // After a successful alloc / before a free.
  virtual void on_alloc(uint64_t base, uint64_t size) = 0;
  virtual void on_free(uint64_t ptr) = 0;
  // Something the history cannot describe (virtual memory management, IPC, a host mapping, the device heap, a reset).
  virtual void on_poison(const char* why) = 0;
};

}  // namespace vgpu::memo
