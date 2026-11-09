// The hash behind launch-level result caching (vgpu/memo.hpp). Four lanes of 64 bits, 32 bytes per step: each lane
// takes one word with a multiply (a bijection of the lane) and a shift-xor, so a changed word always changes its
// lane's state; the lanes are folded at the end with the length. 14 GB/s on one Zen 5 core, 4 KiB pages or megabytes alike
// (docs/perf-design/rounds/10-rounds-sp19-sp27.md, Round sp35), which is what a page costs.
#include "vgpu/memo.hpp"

#include <cstring>

namespace vgpu::memo {

namespace {

struct Lanes {
  uint64_t a, b, c, d;
};

inline Lanes absorb(const uint8_t* p, size_t n) {
  Lanes l{0x243f6a8885a308d3ull, 0x13198a2e03707344ull, 0xa4093822299f31d0ull, 0x082efa98ec4e6c89ull};
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    uint64_t w[4];
    std::memcpy(w, p + i, 32);
    l.a = (l.a ^ w[0]) * 0x9e3779b97f4a7c15ull;
    l.a ^= l.a >> 29;
    l.b = (l.b ^ w[1]) * 0xc2b2ae3d27d4eb4full;
    l.b ^= l.b >> 29;
    l.c = (l.c ^ w[2]) * 0x165667b19e3779f9ull;
    l.c ^= l.c >> 29;
    l.d = (l.d ^ w[3]) * 0x27d4eb2f165667c5ull;
    l.d ^= l.d >> 29;
  }
  // The tail, 8 bytes at a time and then the last few, each into a lane of its own.
  size_t lane = 0;
  uint64_t* ls[4] = {&l.a, &l.b, &l.c, &l.d};
  for (; i + 8 <= n; i += 8, ++lane) {
    uint64_t w;
    std::memcpy(&w, p + i, 8);
    uint64_t& x = *ls[lane & 3];
    x = (x ^ w) * 0x9e3779b97f4a7c15ull;
    x ^= x >> 29;
  }
  if (i < n) {
    uint64_t w = 0;
    std::memcpy(&w, p + i, n - i);
    w ^= uint64_t{n - i} << 56;
    uint64_t& x = *ls[lane & 3];
    x = (x ^ w) * 0x9e3779b97f4a7c15ull;
    x ^= x >> 29;
  }
  return l;
}

}  // namespace

Hash128 hash_bytes(const void* p, size_t n) {
  const Lanes l = absorb(static_cast<const uint8_t*>(p), n);
  Hasher h(0x6d656d6f);
  h.u64(l.a), h.u64(l.b), h.u64(l.c), h.u64(l.d), h.u64(n);
  return h.done();
}

uint64_t page_hash(const uint8_t* p, size_t n) {
  const Lanes l = absorb(p, n);
  return mix64(mix64(l.a ^ (l.b * 0x9e3779b97f4a7c15ull)) ^ mix64(l.c ^ (l.d * 0xc2b2ae3d27d4eb4full)) ^ n);
}

void Hasher::str(const char* s) {
  const size_t n = s ? std::strlen(s) : 0;
  bytes(s, n);
}

}  // namespace vgpu::memo
