// Register residence of the JIT's generated code (src/sass/jit.hpp, RowCtx): random straight-line regions of the generated forms over a
// small set of registers (so rows are reused, written and read again, a destination is often a source, 64-bit pairs overlap), with
// guards (@P, @!P, @UP, @!PT), carries and predicate outputs, and escapes interleaved. Each region is compiled twice, with residence (rows
// kept in locals) and without (every access in the register file, the milestone-2 code), and run on the same random register file,
// predicates, constant bank and group (a whole warp, a part, one lane). The final register file, predicates, counters and everything an
// escape sees (the whole register file and predicates, hashed, at every escape) must be identical; an escape also changes registers and
// predicates, as the tier's handlers do, so a row that is not stored before it, or not reloaded after it, shows. A second run throws from
// an escape in the middle of the region: the register file and counters must then be identical too.
//
// VGPU_JIT_RANDOM_UNITS=<n> (default 4; the rig runs hundreds) units of VGPU_JIT_RANDOM_REGIONS (12) regions each.
// The forms themselves are held to the tier by nvidia/tests/e2e/run_sass_jit.sh; this holds residence to the code without it.
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

#include "sass/jit.hpp"
#include "sass/tier.hpp"
#include "vtest.hpp"

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define VJ_SAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define VJ_SAN 1
#endif
#endif
#ifndef VJ_SAN
#define VJ_SAN 0
#endif

namespace fs = std::filesystem;
using namespace vgpu::sass;
using tier::Form;

namespace {

constexpr unsigned kRegs = 16;   // rows of the register file
constexpr uint16_t kNo = 0xffff;

struct Gen {
  std::mt19937_64 rng;
  explicit Gen(uint64_t seed) : rng(seed) {}
  uint64_t u(uint64_t n) { return rng() % n; }
  bool coin(unsigned pct) { return u(100) < pct; }

  uint32_t special() {
    static const uint32_t v[] = {0u, 1u, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x00000001u, 0x80000001u,
                                 0x3f800000u, 0xbf800000u, 0x00400000u, 31u, 32u, 63u, 64u};
    return coin(50) ? v[u(sizeof v / sizeof v[0])] : static_cast<uint32_t>(rng());
  }
  // A 32-bit source.
  jit::SrcFacts src(bool flt) {
    jit::SrcFacts s;
    const unsigned k = static_cast<unsigned>(u(100));
    if (k < 62) {
      s.kind = 0;
      s.off = static_cast<uint32_t>(u(kRegs)) * 32u;
    } else if (k < 72) {
      s.kind = 1;
      s.val = special();
    } else if (k < 86) {
      s.kind = 2;
      s.dyn = 0;
      s.off = static_cast<uint32_t>(u(kRegs));
      s.mod = static_cast<uint8_t>(flt ? (u(4) | (coin(30) ? 4 : 0)) : u(4));
    } else if (k < 93) {
      s.kind = 2;
      s.dyn = 1;
      s.off = static_cast<uint32_t>(u(8)) * 4u;
      s.mod = static_cast<uint8_t>(flt ? u(4) : u(4));
    } else {
      s.kind = 2;
      s.dyn = 3;
      s.off = static_cast<uint32_t>(u(32));
      s.mod = static_cast<uint8_t>(u(4));
    }
    return s;
  }
  // A 64-bit source: a register pair, a constant pair, bank 0 or uniform registers as a pair.
  jit::SrcFacts src64() {
    jit::SrcFacts s;
    const unsigned k = static_cast<unsigned>(u(100));
    if (k < 60) {
      s.kind = 0;
      s.off = static_cast<uint32_t>(u(kRegs - 1)) * 32u;
    } else if (k < 75) {
      s.kind = 1;
      s.val = special();
      s.val2 = special();
    } else if (k < 88) {
      s.kind = 2;
      s.dyn = 1;
      s.off = static_cast<uint32_t>(u(7)) * 4u;
    } else {
      s.kind = 2;
      s.dyn = 3;
      s.off = static_cast<uint32_t>(u(30));
    }
    s.mod = static_cast<uint8_t>(u(4));
    return s;
  }
  jit::SrcFacts pred() {
    jit::SrcFacts s;
    s.kind = 2;
    s.dyn = coin(80) ? 0 : 3;
    s.off = static_cast<uint32_t>(u(8));
    s.mod = coin(30) ? 1 : 0;
    return s;
  }
  uint16_t dreg() { return coin(5) ? kNo : static_cast<uint16_t>(u(kRegs - 1)); }
  uint16_t dpred(bool allow_no) { return allow_no && coin(30) ? kNo : static_cast<uint16_t>(u(8)); }

  jit::OpFacts op() {
    jit::OpFacts o;
    o.lo = rng();
    o.hi = rng();
    o.cls = static_cast<uint8_t>(u(12));
    if (coin(55)) {
      o.unguarded = true;
    } else {
      o.unguarded = false;
      o.guard = static_cast<uint8_t>(u(8));
      o.guard_not = coin(40);
      o.guard_uniform = coin(25);
    }
    if (coin(14)) return o;   // an escape
    static const Form forms[] = {Form::Mov,   Form::Iadd3,    Form::Iadd3x, Form::Imad,   Form::Imad32, Form::ImadWide, Form::ImadHi, Form::Lop3,
                                 Form::Lea,   Form::Shf,      Form::Isetp,  Form::Fsetp,  Form::Fsel,   Form::Ffma,     Form::Fadd,   Form::Fmul,
                                 Form::Prmt,  Form::Imnmx,    Form::I2fp,   Form::Sel,    Form::Plop3,    Form::S2r,    Form::Cs2r,   Form::Umov,
                                 Form::Uldc,  Form::Uiadd3,   Form::Ulop3,  Form::Usel,  Form::Uisetp};
    const Form f = forms[u(sizeof forms / sizeof forms[0])];
    o.form = static_cast<uint8_t>(f);
    o.dst = dreg();
    switch (f) {
      case Form::Mov: o.s[0] = src(false); break;
      case Form::Iadd3: for (auto& x : o.s) x = src(false); o.a = static_cast<uint32_t>(u(4)); break;
      case Form::Iadd3x:
        for (auto& x : o.s) x = src(false);
        o.a = static_cast<uint32_t>(u(4));
        o.aux = coin(50) ? 1 : 0;
        o.p = pred();
        o.p2 = pred();
        o.dst2 = dpred(true);
        o.c = coin(40) ? kNo : static_cast<uint32_t>(u(8));
        break;
      case Form::Imad: for (auto& x : o.s) x = src(false); break;
      case Form::Imad32:
        for (auto& x : o.s) x = src(false);
        o.aux = coin(50) ? 1 : 0;
        o.p = pred();
        o.dst2 = dpred(true);
        break;
      case Form::ImadWide:
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.s[2] = src64();
        o.aux = static_cast<uint8_t>(u(4));
        o.p = pred();
        o.dst2 = dpred(true);
        break;
      case Form::ImadHi: {
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.aux = static_cast<uint8_t>(u(8));
        o.s[2] = (o.aux & 4) ? src64() : src(false);
        o.p = pred();
        o.dst2 = dpred(true);
        break;
      }
      case Form::Lop3:
        for (auto& x : o.s) x = src(false);
        o.a = static_cast<uint32_t>(u(256));
        o.aux = coin(60) ? 0 : static_cast<uint8_t>(1 | (coin(50) ? 2 : 0));
        o.p = pred();
        o.dst2 = dpred(false);
        break;
      case Form::Lea:
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.s[2] = src(false);
        o.a = static_cast<uint32_t>(u(34));
        o.aux = static_cast<uint8_t>(u(16));
        o.p = pred();
        o.dst2 = dpred(true);
        break;
      case Form::Shf:
        for (auto& x : o.s) x = src(false);
        o.a = static_cast<uint32_t>(u(4));
        o.aux = static_cast<uint8_t>(u(8));
        break;
      case Form::Isetp:
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.a = static_cast<uint32_t>(1 + u(6));
        o.b = static_cast<uint32_t>(u(3));
        o.c = static_cast<uint32_t>(u(2));
        o.p = pred();
        o.dst = static_cast<uint16_t>(u(8));
        o.dst2 = static_cast<uint16_t>(u(8));
        break;
      case Form::Fsetp:
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.a = static_cast<uint32_t>(u(16));
        o.b = static_cast<uint32_t>(u(3));
        o.c = static_cast<uint32_t>(u(2));
        o.p = pred();
        o.dst = static_cast<uint16_t>(u(8));
        o.dst2 = static_cast<uint16_t>(u(8));
        break;
      case Form::Fsel:
        o.s[0] = src(true);
        o.s[1] = src(true);
        o.p = pred();
        break;
      case Form::Ffma: for (auto& x : o.s) x = src(true); o.a = static_cast<uint32_t>(u(4)); break;
      case Form::Fadd: case Form::Fmul: o.s[0] = src(true); o.s[1] = src(true); o.a = static_cast<uint32_t>(u(4)); break;
      case Form::Prmt: for (auto& x : o.s) x = src(false); break;
      case Form::Imnmx: o.s[0] = src(false); o.s[1] = src(false); o.c = static_cast<uint32_t>(u(2)); o.p = pred(); break;
      case Form::I2fp: o.s[0] = src(false); o.c = static_cast<uint32_t>(u(2)); break;
      case Form::Sel: o.s[0] = src(false); o.s[1] = src(false); o.p = pred(); break;
      case Form::Plop3:
        for (auto& x : o.s) x = pred();
        o.a = static_cast<uint32_t>(u(256));
        o.b = static_cast<uint32_t>(u(256));
        o.dst = static_cast<uint16_t>(u(8));
        o.dst2 = static_cast<uint16_t>(u(8));
        break;
      case Form::S2r: o.a = static_cast<uint32_t>(u(24)); break;
      case Form::Cs2r: o.a = coin(80) ? 0xffu : static_cast<uint32_t>(u(8)); o.b = coin(50) ? 1 : 0; break;
      case Form::Umov: o.s[0] = src(false); o.dst = coin(10) ? kNo : static_cast<uint16_t>(u(40)); break;
      case Form::Uldc:
        o.a = coin(90) ? 0 : 3;   // (bank 3 is the handler's)
        o.b = static_cast<uint32_t>(u(6));
        o.imm = static_cast<int64_t>(u(56));
        o.dst = coin(10) ? kNo : static_cast<uint16_t>(u(40));
        o.c = static_cast<uint32_t>(u(44));
        break;
      case Form::Uiadd3:
        for (auto& x : o.s) x = src(false);
        o.b = static_cast<uint32_t>(u(64));
        o.dst = coin(10) ? kNo : static_cast<uint16_t>(u(40));
        o.aux = coin(40) ? 1 : 0;
        o.p = pred();
        o.p2 = pred();
        o.dst2 = coin(40) ? kNo : static_cast<uint16_t>(u(8));
        o.a = coin(50) ? kNo : static_cast<uint32_t>(u(8));
        if (o.dst2 != kNo || coin(30)) o.aux |= 2;
        if (o.a != kNo || coin(30)) o.aux |= 4;
        break;
      case Form::Ulop3: for (auto& x : o.s) x = src(false); o.a = static_cast<uint32_t>(u(256)); o.dst = coin(10) ? kNo : static_cast<uint16_t>(u(40)); break;
      case Form::Usel: o.s[0] = src(false); o.s[1] = src(false); o.p = pred(); o.dst = coin(10) ? kNo : static_cast<uint16_t>(u(40)); break;
      case Form::Uisetp:
        o.s[0] = src(false);
        o.s[1] = src(false);
        o.a = static_cast<uint32_t>(u(8));
        o.b = static_cast<uint32_t>(u(3));
        o.c = static_cast<uint32_t>(u(2));
        o.p = pred();
        o.dst = static_cast<uint16_t>(u(8));
        o.dst2 = static_cast<uint16_t>(u(8));
        break;
      default: break;
    }
    if (!jit::generates(o)) {   // (an escape, if the generator will not take it)
      jit::OpFacts e;
      e.lo = o.lo; e.hi = o.hi; e.cls = o.cls; e.guard = o.guard; e.guard_not = o.guard_not; e.guard_uniform = o.guard_uniform; e.unguarded = o.unguarded;
      return e;
    }
    return o;
  }
};

// The machine a region runs on, and what an escape does to it.
struct Sim {
  alignas(32) uint32_t R[kRegs * 32];
  uint32_t p[8];
  bool up[8];
  uint32_t ur[256];
  uint8_t b0[64];
  uint64_t steps = 0, insts = 0, tinsts = 0, by_class[16] = {};
  bool gate = false;   // the warp's store gate
  uint64_t seed = 0;
  std::vector<uint64_t> log;
  long throw_at = -1;
  long calls = 0;
  const tier::MOp* base = nullptr;   // the region's MOps (their handlers are the trampolines below): an op's number is its place among them
  void (*escape)(jit::Ctx*, uint32_t, uint32_t) = nullptr;
};
// The tier's handler of an escape, as the generated code calls it (Ctx::ops, Ctx::tctx): it works out which op this is and runs the test's own effect.
inline void sim_handler(void* ctx, const tier::MOp& m, uint32_t ex) {
  Sim& s = *static_cast<Sim*>(ctx);
  jit::Ctx tmp{};
  tmp.host = &s;
  s.escape(&tmp, static_cast<uint32_t>(&m - s.base), ex);
}

uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  h *= 0xff51afd7ed558ccdull;
  return h ^ (h >> 33);
}

void sim_escape(jit::Ctx* c, uint32_t op, uint32_t ex) {
  Sim& s = *static_cast<Sim*>(c->host);
  uint64_t h = mix(s.seed, op * 977u + ex);
  for (unsigned i = 0; i < kRegs * 32; ++i) h = mix(h, s.R[i]);
  for (unsigned i = 0; i < 8; ++i) h = mix(h, s.p[i]);
  s.log.push_back(h);
  if (s.throw_at == s.calls++) throw std::runtime_error("escape fault");
  // The handler's effect: some rows (any of them: it is a handler the generator knows nothing of), for the executing lanes, and a predicate.
  std::mt19937_64 r(h);
  const unsigned nrows = static_cast<unsigned>(r() % 4);
  for (unsigned k = 0; k < nrows; ++k) {
    const unsigned row = static_cast<unsigned>(r() % kRegs);
    const uint32_t v = static_cast<uint32_t>(r());
    for (unsigned l = 0; l < 32; ++l)
      if ((ex >> l) & 1u) s.R[row * 32 + l] = v ^ (l * 0x9e3779b1u);
  }
  if (r() % 3 == 0) {
    const unsigned pr = static_cast<unsigned>(r() % 7);
    s.p[pr] = (s.p[pr] & ~ex) | (static_cast<uint32_t>(r()) & ex);
  }
}
void sim_note(jit::Ctx*, uint32_t) {}
// S2R's host function here: the lanes' special register is a function of its number and the lane.
void sim_sreg(jit::Ctx*, uint32_t sr, uint32_t* d, uint32_t ex) {
  for (unsigned l = 0; l < 32; ++l)
    if ((ex >> l) & 1u) d[l] = sr * 0x01000193u + l * 0x9e3779b1u;
}

// The MOps a region's ops are called through: as many as the largest region, every handler the trampoline.
const std::vector<tier::MOp>& test_mops() {
  static const std::vector<tier::MOp> v = [] {
    std::vector<tier::MOp> m(512);
    for (tier::MOp& x : m) x.fn = &sim_handler;
    return m;
  }();
  return v;
}

jit::Ctx make_ctx(Sim& s, bool stats) {
  s.base = test_mops().data();
  s.escape = &sim_escape;
  jit::Ctx c{};
  c.abi = jit::kAbiVersion;
  c.flags = stats ? 1u : 0u;
  c.up = s.up;
  c.p = s.p;
  c.ur = s.ur;
  c.steps = &s.steps;
  c.instructions = &s.insts;
  c.thread_instructions = &s.tinsts;
  c.by_class = s.by_class;
  c.tctx = &s;
  c.gate = &s.gate;
  c.ops = reinterpret_cast<const uint8_t*>(test_mops().data());
  c.last = ~0u;
  c.note = &sim_note;
  c.sreg = &sim_sreg;
  c.host = &s;
  c.R = s.R;
  c.b0 = s.b0;
  c.b0n = sizeof s.b0;
  c.max_steps = ~uint64_t{0};
  return c;
}

void randomize(Sim& s, std::mt19937_64& rng) {
  for (auto& v : s.R) v = static_cast<uint32_t>(rng());
  for (auto& v : s.p) v = static_cast<uint32_t>(rng());
  s.p[7] = 0xffffffffu;
  for (auto& v : s.up) v = rng() & 1;
  s.up[7] = true;
  for (auto& v : s.ur) v = static_cast<uint32_t>(rng());
  for (auto& v : s.b0) v = static_cast<uint8_t>(rng());
  s.steps = s.insts = s.tinsts = 0;
  std::memset(s.by_class, 0, sizeof s.by_class);
  s.log.clear();
  s.calls = 0;
}

bool same(const Sim& a, const Sim& b) {
  return std::memcmp(a.R, b.R, sizeof a.R) == 0 && std::memcmp(a.p, b.p, sizeof a.p) == 0 && std::memcmp(a.up, b.up, sizeof a.up) == 0 && std::memcmp(a.ur, b.ur, sizeof a.ur) == 0 && a.steps == b.steps && a.insts == b.insts &&
         a.tinsts == b.tinsts && std::memcmp(a.by_class, b.by_class, sizeof a.by_class) == 0 && a.log == b.log;
}

}  // namespace

VTEST(jit_residence_random_regions_equal_the_register_file_code) {
  if (VJ_SAN) return;
  {
    jit::Config probe;
    jit::Engine e(probe);
    if (e.compiler_id().empty()) {
      std::printf("  (no C++ compiler on PATH: skipped)\n");
      return;
    }
  }
  const unsigned units = static_cast<unsigned>(std::strtoul(std::getenv("VGPU_JIT_RANDOM_UNITS") ? std::getenv("VGPU_JIT_RANDOM_UNITS") : "4", nullptr, 10));
  const unsigned nreg = static_cast<unsigned>(std::strtoul(std::getenv("VGPU_JIT_RANDOM_REGIONS") ? std::getenv("VGPU_JIT_RANDOM_REGIONS") : "12", nullptr, 10));
  const uint64_t seed0 = std::getenv("VGPU_JIT_RANDOM_SEED") ? std::strtoull(std::getenv("VGPU_JIT_RANDOM_SEED"), nullptr, 10) : 1;
  const char* t = std::getenv("TMPDIR");
  std::string tmpl = std::string(t && *t ? t : "/tmp") + "/vgpu-test-jit-resid.XXXXXX";
  std::vector<char> b(tmpl.begin(), tmpl.end());
  b.push_back(0);
  if (!::mkdtemp(b.data())) throw vtest::Failure("mkdtemp failed");
  const std::string dir = b.data();
  std::atomic<unsigned> next{0}, failures{0}, regions_checked{0}, runs{0}, escapes_seen{0}, gen_ops{0};
  std::mutex mu;
  std::string first_failure;
  const auto worker = [&] {
    for (;;) {
      const unsigned unit = next.fetch_add(1);
      if (unit >= units) return;
      Gen g(seed0 * 1000003u + unit);
      jit::UnitSpec spec;
      spec.sm = 80;
      for (unsigned r = 0; r < nreg; ++r) {
        jit::RegionSpec rs;
        const unsigned n = 4 + static_cast<unsigned>(g.u(36));
        for (unsigned i = 0; i < n; ++i) {
          rs.ops.push_back(g.op());
          if (rs.ops.back().form) ++gen_ops;
        }
        spec.regions.push_back(std::move(rs));
      }
      jit::Config on, off;
      on.cache_dir = dir + "/on";
      off.cache_dir = dir + "/off";
      off.residence = false;
      jit::Engine eon(on), eoff(off);
      const jit::Result ron = eon.obtain(spec), roff = eoff.obtain(spec);
      if (!ron.found || !roff.found) {
        std::lock_guard<std::mutex> lk(mu);
        ++failures;
        if (first_failure.empty()) first_failure = "unit " + std::to_string(unit) + ": compile failed: " + (ron.found ? roff.message : ron.message);
        continue;
      }
      std::mt19937_64 rng(seed0 * 7919u + unit);
      for (unsigned r = 0; r < nreg; ++r) {
        ++regions_checked;
        const uint32_t glanes_of[] = {32, 32, 32};
        (void)glanes_of;
        for (unsigned trial = 0; trial < 6; ++trial) {
          Sim a, bsim;
          randomize(a, rng);
          uint32_t group = trial < 2 ? 0xffffffffu : trial == 2 ? 0x0000ffffu : trial == 3 ? 0x00000001u : static_cast<uint32_t>(rng()) | 1u;
          if (trial == 5) group = 0xffffu << 16;   // the second half of a warp (a 48-thread block's last warp is the first half)
          bsim = a;
          a.seed = bsim.seed = rng();
          const bool stats = trial & 1;
          const bool throw_case = trial == 4 && r % 2 == 0;
          if (throw_case) a.throw_at = bsim.throw_at = static_cast<long>(rng() % 3);
          jit::Ctx ca = make_ctx(a, stats), cb = make_ctx(bsim, stats);
          bool ta = false, tb = false;
          const uint32_t gl = static_cast<uint32_t>(__builtin_popcount(group));
          try { ron.lib->fns[r](&ca, group, gl, 0, 0); } catch (const std::runtime_error&) { ta = true; }
          try { roff.lib->fns[r](&cb, group, gl, 0, 0); } catch (const std::runtime_error&) { tb = true; }
          ++runs;
          escapes_seen += static_cast<unsigned>(a.log.size());
          if (ta != tb || !same(a, bsim)) {
            std::lock_guard<std::mutex> lk(mu);
            ++failures;
            if (first_failure.empty()) {
              first_failure = "unit " + std::to_string(unit) + " region " + std::to_string(r) + " trial " + std::to_string(trial) + ": differs (R " +
                              (std::memcmp(a.R, bsim.R, sizeof a.R) ? "differs" : "same") + ", p " + (std::memcmp(a.p, bsim.p, sizeof a.p) ? "differs" : "same") +
                              ", counters " + (a.insts == bsim.insts && a.tinsts == bsim.tinsts ? "same" : "differ") + ", escape logs " + (a.log == bsim.log ? "same" : "differ") +
                              ", threw " + std::to_string(ta) + "/" + std::to_string(tb) + ")";
            }
          }
        }
      }
    }
  };
  const unsigned nthreads = std::min<unsigned>(units, std::max(1u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < nthreads; ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  std::printf("  residence: %u units, %u regions, %u runs, %u generated ops, %u escapes executed, %u failures\n", units, regions_checked.load(), runs.load(),
              gen_ops.load(), escapes_seen.load(), failures.load());
  std::error_code ec;
  fs::remove_all(dir, ec);
  if (failures) throw vtest::Failure(first_failure);
}

// ---- control flow in generated code (milestone 5) ------------------------------------------------------------------------------------------------
//
// Regions with branches and cont ops. Two checks: (1) a region of escapes, branches and cont ops against a model of the run loop's own rules
// (tier_ops.inc: the cont path -- a branch the whole group takes or none of it, to a place in the section, the loop going on only while the chain's
// conditions hold and the turn has a call left; the 256-op run bound, the step budget, the counters), written here from those rules and not from
// the generator; (2) a region that also holds generated ALU ops (residence across labels and branches) against the same region compiled without
// residence. Every trial also checks what the generated code reports on the way out: the op the loop runs next and where its 256-op run restarted.

namespace {

struct GSim : Sim {
  uint32_t budget = 0, extra = 0;
  uint64_t divergent = 0, max_steps = ~uint64_t{0};
  int32_t exit_idx = -1, exit_ps = 0;
  uint64_t visit = 0;
};

// The handler of an escape here: registers and predicates change as a handler's would; the predicates mostly become "no lane" or "every lane", so
// that branches are uniform often enough to be taken.
void graph_escape(jit::Ctx* c, uint32_t op, uint32_t ex) {
  GSim& s = *static_cast<GSim*>(c->host);
  uint64_t h = mix(s.seed, op * 977u + ex + 31u * s.visit++);
  for (unsigned i = 0; i < kRegs * 32; ++i) h = mix(h, s.R[i]);
  for (unsigned i = 0; i < 8; ++i) h = mix(h, s.p[i]);
  s.log.push_back(h);
  if (s.throw_at == s.calls++) throw std::runtime_error("escape fault");
  std::mt19937_64 r(h);
  const unsigned nrows = static_cast<unsigned>(r() % 3);
  for (unsigned k = 0; k < nrows; ++k) {
    const unsigned row = static_cast<unsigned>(r() % kRegs);
    const uint32_t v = static_cast<uint32_t>(r());
    for (unsigned l = 0; l < 32; ++l)
      if ((ex >> l) & 1u) s.R[row * 32 + l] = v ^ (l * 0x9e3779b1u);
  }
  for (unsigned k = 0; k < 2; ++k) {
    const unsigned pr = static_cast<unsigned>(r() % 7);
    const unsigned how = static_cast<unsigned>(r() % 5);
    const uint32_t v = how < 2 ? 0u : how < 4 ? 0xffffffffu : static_cast<uint32_t>(r());
    s.p[pr] = (s.p[pr] & ~ex) | (v & ex);
  }
}

jit::OpFacts graph_base(Gen& g) {
  jit::OpFacts o;
  o.lo = g.rng();
  o.hi = g.rng();
  o.cls = static_cast<uint8_t>(g.u(12));
  if (g.coin(60)) {
    o.unguarded = true;
  } else {
    o.unguarded = false;
    o.guard = static_cast<uint8_t>(g.u(8));
    o.guard_not = g.coin(40);
    o.guard_uniform = g.coin(25);
  }
  return o;
}
jit::OpFacts graph_ctl(Gen& g, unsigned n, bool cont) {
  jit::OpFacts o = graph_base(g);
  if (cont) {
    o.cont = true;
    return o;
  }
  o.br = true;
  o.aux = static_cast<uint8_t>(g.u(3));
  o.p = g.pred();
  o.p2 = g.pred();
  o.tgt = static_cast<int32_t>(g.u(n + 4)) - 2;   // a little outside the region on both sides too
  return o;
}

// A region of n ops: escapes (or, mixed, the generated forms and escapes of Gen::op), with branches and cont ops among them.
jit::RegionSpec graph_region(Gen& g, unsigned n, bool mixed, unsigned ctl_pct) {
  jit::RegionSpec rs;
  for (unsigned i = 0; i < n; ++i) {
    const unsigned k = static_cast<unsigned>(g.u(100));
    if (k < ctl_pct) {
      rs.ops.push_back(graph_ctl(g, n, g.coin(30)));
    } else if (!mixed && g.coin(10)) {   // a store (an escape): the generated code leaves before it while the warp's store gate is up
      jit::OpFacts o = graph_base(g);
      o.store = true;
      rs.ops.push_back(o);
    }
    else if (mixed) {
      jit::OpFacts o = g.op();
      // The forms the tier (and the JIT) go on past are not straight: S2R, SEL, PLOP3 and the uniform datapath.
      switch (static_cast<Form>(o.form)) {
        case Form::S2r: case Form::Sel: case Form::Plop3: case Form::Umov: case Form::Uldc: case Form::Usel: case Form::Uiadd3: case Form::Ulop3: case Form::Uisetp: case Form::Cs2r:
          o.cont = g.coin(80);
          break;
        default:
          if (o.form == 0) o.store = g.coin(15);   // (an escape that is a store)
          break;
      }
      rs.ops.push_back(o);
    }
    else rs.ops.push_back(graph_base(g));
  }
  return rs;
}

uint32_t ex_of(const jit::OpFacts& o, uint32_t group, const GSim& s) {
  if (o.unguarded || (o.guard == 7 && !o.guard_not)) return group;
  uint32_t m;
  if (o.guard_uniform) m = (o.guard == 7 || s.up[o.guard]) ? 0xffffffffu : 0u;
  else m = o.guard == 7 ? 0xffffffffu : s.p[o.guard];
  if (o.guard_not) m = ~m;
  return group & m;
}
uint32_t pm_of(const jit::SrcFacts& x, const GSim& s) {
  uint32_t m;
  if (x.dyn == 3) m = (x.off == 7 || s.up[x.off]) ? 0xffffffffu : 0u;
  else m = x.off == 7 ? 0xffffffffu : s.p[x.off];
  return (x.mod & 1u) ? ~m : m;
}

// The run loop, for the ops of one region (see the comment above).
void model_graph(const jit::RegionSpec& r, GSim& s, uint32_t group, uint32_t entry, uint32_t n0, uint32_t flags) {
  const uint32_t k = static_cast<uint32_t>(r.ops.size());
  const bool chain = flags & jit::kFlagChain;
  jit::Ctx host{};
  host.host = &s;
  int32_t ps = jit::kNoPassStart;
  const auto leave = [&](int32_t idx) { s.exit_idx = idx; s.exit_ps = ps; };
  const auto count = [&](const jit::OpFacts& o, uint32_t ex) {
    ++s.steps;
    ++s.insts;
    s.tinsts += static_cast<uint32_t>(__builtin_popcount(ex));
    s.by_class[o.cls] += static_cast<uint32_t>(__builtin_popcount(ex));
  };
  const std::vector<uint32_t> starts = jit::stretch_starts(r);
  if (std::find(starts.begin(), starts.end(), entry) == starts.end()) {
    leave(static_cast<int32_t>(entry));
    return;
  }
  uint32_t pos = entry, n = n0;
  for (;;) {
    const uint32_t e = jit::stretch_end(r, pos);
    uint32_t straight = 0;
    for (uint32_t j = pos; j < e; ++j) straight += jit::is_ctl(r.ops[j]) ? 0u : 1u;
    // what the loop checks for each op of the stretch: the run bound counts the straight ops; the step budget counts all
    if (n + straight > 256u || s.steps + (e - pos) > s.max_steps) {
      leave(static_cast<int32_t>(pos));
      return;
    }
    bool moved = false;
    for (uint32_t i = pos; i < e; ++i) {
      const jit::OpFacts& o = r.ops[i];
      const uint32_t ex = ex_of(o, group, s);
      if (o.br) {
        uint32_t take = ex;
        if (o.aux) take &= pm_of(o.p, s);
        if (o.aux > 1) take &= pm_of(o.p2, s);
        if (!chain || s.budget == 0 || (take != group && take != 0)) {
          leave(static_cast<int32_t>(i));
          return;
        }
        count(o, ex);
        if (ex && ex != group) ++s.divergent;
        --s.budget;
        ++s.extra;
        n = 0;
        const int32_t next = take ? o.tgt : static_cast<int32_t>(i) + 1;
        ps = next;
        if (next < 0 || static_cast<uint32_t>(next) >= k) {
          leave(next);
          return;
        }
        pos = static_cast<uint32_t>(next);
        moved = true;
        break;
      }
      if (o.cont) {
        if (!chain || s.budget == 0) {
          leave(static_cast<int32_t>(i));
          return;
        }
        count(o, ex);
        if (ex) graph_escape(&host, i, ex);
        --s.budget;
        ++s.extra;
        n = 0;
        ps = static_cast<int32_t>(i) + 1;
        if (i + 1 >= k) {
          leave(static_cast<int32_t>(k));
          return;
        }
        pos = i + 1;
        moved = true;
        break;
      }
      if (o.store && s.gate) {   // the loop's store gate: the loop ends the warp's turn at this op, once
        leave(static_cast<int32_t>(i));
        return;
      }
      count(o, ex);
      if (ex) graph_escape(&host, i, ex);
    }
    if (!moved) {
      leave(static_cast<int32_t>(k));
      return;
    }
  }
}

jit::Ctx make_gctx(GSim& s, uint32_t flags) {
  jit::Ctx c = make_ctx(s, false);
  s.escape = &graph_escape;
  c.flags = flags;
  c.divergent = &s.divergent;
  c.budget = &s.budget;
  c.extra = &s.extra;
  c.max_steps = s.max_steps;
  return c;
}

void randomize_g(GSim& s, std::mt19937_64& rng, bool long_region) {
  randomize(s, rng);
  static const uint32_t budgets[] = {0, 1, 2, 3, 5, 20, 200};
  s.budget = budgets[rng() % (sizeof budgets / sizeof budgets[0])];
  s.extra = static_cast<uint32_t>(rng() % 5);
  s.divergent = rng() % 7;
  s.steps = rng() % 50;
  // (Always finite: a loop the generated code runs on past its turn's calls -- a mutant that drops the budget -- ends at the step limit and the state differs from the model's, instead of never ending.)
  s.max_steps = rng() % 4 == 0 ? s.steps + rng() % 120 : s.steps + 300 + rng() % 5000;
  s.visit = 0;
  s.gate = rng() % 3 == 0;
  (void)long_region;
}

bool same_g(const GSim& a, const GSim& b, bool compare_exit) {
  return same(a, b) && a.budget == b.budget && a.extra == b.extra && a.divergent == b.divergent && (!compare_exit || (a.exit_idx == b.exit_idx && a.exit_ps == b.exit_ps));
}

bool have_cxx() {
  jit::Config probe;
  jit::Engine e(probe);
  if (e.compiler_id().empty()) {
    std::printf("  (no C++ compiler on PATH: skipped)\n");
    return false;
  }
  return true;
}

std::string tmp_dir(const char* tag) {
  const char* t = std::getenv("TMPDIR");
  std::string tmpl = std::string(t && *t ? t : "/tmp") + "/vgpu-test-jit-" + tag + ".XXXXXX";
  std::vector<char> b(tmpl.begin(), tmpl.end());
  b.push_back(0);
  if (!::mkdtemp(b.data())) throw vtest::Failure("mkdtemp failed");
  return b.data();
}

// How the trial's group, entry and loop state are drawn.
struct Trial {
  uint32_t group, entry, n0, flags;
};
Trial draw_trial(std::mt19937_64& rng, const jit::RegionSpec& rs, unsigned t) {
  Trial tr;
  static const uint32_t groups[] = {0xffffffffu, 0x1u, 0x80000000u, 0x00030000u, 0x0000ffffu};
  tr.group = t % 5 == 4 ? (static_cast<uint32_t>(rng()) | 1u) : groups[t % 5];
  const std::vector<uint32_t> st = jit::stretch_starts(rs);
  tr.entry = st[rng() % st.size()];
  if (rng() % 40 == 0) tr.entry = static_cast<uint32_t>(rng() % rs.ops.size());   // (not a stretch start: the function leaves at once)
  static const uint32_t n0s[] = {0, 0, 0, 0, 100, 240, 250, 255, 256, 300};
  tr.n0 = n0s[rng() % (sizeof n0s / sizeof n0s[0])];
  tr.flags = (rng() % 4 ? jit::kFlagChain : 0u) | (rng() % 4 == 0 ? jit::kFlagTierStats : 0u);
  return tr;
}

}  // namespace

VTEST(jit_graph_regions_equal_the_model_of_the_run_loop) {
  if (VJ_SAN || !have_cxx()) return;
  const unsigned units = static_cast<unsigned>(std::strtoul(std::getenv("VGPU_JIT_GRAPH_UNITS") ? std::getenv("VGPU_JIT_GRAPH_UNITS") : "6", nullptr, 10));
  const uint64_t seed0 = std::getenv("VGPU_JIT_RANDOM_SEED") ? std::strtoull(std::getenv("VGPU_JIT_RANDOM_SEED"), nullptr, 10) : 1;
  const std::string dir = tmp_dir("graph");
  std::atomic<unsigned> next{0}, failures{0}, runs{0}, exits_in{0}, exits_out{0}, throws{0}, taken{0};
  std::mutex mu;
  std::string first_failure;
  const auto worker = [&] {
    for (;;) {
      const unsigned unit = next.fetch_add(1);
      if (unit >= units) return;
      Gen g(seed0 * 7000003u + unit);
      jit::UnitSpec spec;
      spec.sm = 80;
      for (unsigned r = 0; r < 12; ++r) {
        const bool lng = r % 6 == 5;   // a long region with few branches: the 256-op bound comes into play
        spec.regions.push_back(graph_region(g, lng ? 150 + static_cast<unsigned>(g.u(70)) : 4 + static_cast<unsigned>(g.u(40)), false, lng ? 3 : 25));
      }
      jit::Config cfg;
      cfg.cache_dir = dir + "/u" + std::to_string(unit);
      jit::Engine eng(cfg);
      const jit::Result res = eng.obtain(spec);
      if (!res.found) {
        std::lock_guard<std::mutex> lk(mu);
        ++failures;
        if (first_failure.empty()) first_failure = "unit " + std::to_string(unit) + ": compile failed: " + res.message;
        continue;
      }
      std::mt19937_64 rng(seed0 * 104729u + unit);
      for (unsigned r = 0; r < spec.regions.size(); ++r) {
        for (unsigned t = 0; t < 40; ++t) {
          GSim a;
          randomize_g(a, rng, false);
          a.seed = rng();
          const Trial tr = draw_trial(rng, spec.regions[r], t);
          if (t % 7 == 3) a.throw_at = static_cast<long>(rng() % 4);
          GSim m = a;
          jit::Ctx ca = make_gctx(a, tr.flags);
          bool ta = false, tm = false;
          try { res.lib->fns[r](&ca, tr.group, static_cast<uint32_t>(__builtin_popcount(tr.group)), tr.entry, tr.n0); } catch (const std::runtime_error&) { ta = true; }
          a.exit_idx = ca.exit_idx;
          a.exit_ps = ca.exit_ps;
          try { model_graph(spec.regions[r], m, tr.group, tr.entry, tr.n0, tr.flags); } catch (const std::runtime_error&) { tm = true; }
          ++runs;
          if (ta) ++throws;
          if (!ta && a.exit_idx >= 0 && static_cast<size_t>(a.exit_idx) < spec.regions[r].ops.size()) ++exits_in;
          else if (!ta) ++exits_out;
          taken += a.extra > 0 ? 1u : 0u;
          if (ta != tm || !same_g(a, m, !ta)) {
            std::lock_guard<std::mutex> lk(mu);
            ++failures;
            if (first_failure.empty())
              first_failure = "unit " + std::to_string(unit) + " region " + std::to_string(r) + " trial " + std::to_string(t) + " (group " + std::to_string(tr.group) + ", entry " +
                              std::to_string(tr.entry) + ", n0 " + std::to_string(tr.n0) + ", flags " + std::to_string(tr.flags) + ", budget " + std::to_string(m.budget) +
                              "): differs from the model: exit " + std::to_string(a.exit_idx) + "/" + std::to_string(m.exit_idx) + " ps " + std::to_string(a.exit_ps) + "/" +
                              std::to_string(m.exit_ps) + " R " + (std::memcmp(a.R, m.R, sizeof a.R) ? "differs" : "same") + " p " +
                              (std::memcmp(a.p, m.p, sizeof a.p) ? "differs" : "same") + " steps " + std::to_string(a.steps) + "/" + std::to_string(m.steps) + " insts " +
                              std::to_string(a.insts) + "/" + std::to_string(m.insts) + " budget " + std::to_string(a.budget) + "/" + std::to_string(m.budget) + " extra " +
                              std::to_string(a.extra) + "/" + std::to_string(m.extra) + " divergent " + std::to_string(a.divergent) + "/" + std::to_string(m.divergent) +
                              " log " + (a.log == m.log ? "same" : "differs") + " threw " + std::to_string(ta) + "/" + std::to_string(tm);
          }
        }
      }
    }
  };
  const unsigned nthreads = std::min<unsigned>(units, std::max(1u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < nthreads; ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  std::printf("  graph: %u units, %u runs, %u left inside the region, %u outside it, %u with a branch or cont op taken, %u threw, %u failures\n", units, runs.load(),
              exits_in.load(), exits_out.load(), taken.load(), throws.load(), failures.load());
  std::error_code ec;
  fs::remove_all(dir, ec);
  if (failures) throw vtest::Failure(first_failure);
  // The trials must have exercised what they claim to: loops taken, regions left both ways.
  VCHECK(taken.load() > runs.load() / 10);
  VCHECK(exits_in.load() > runs.load() / 20);
  VCHECK(exits_out.load() > runs.load() / 20);
}

VTEST(jit_graph_regions_with_residence_equal_the_register_file_code) {
  if (VJ_SAN || !have_cxx()) return;
  const unsigned units = static_cast<unsigned>(std::strtoul(std::getenv("VGPU_JIT_GRAPH_UNITS") ? std::getenv("VGPU_JIT_GRAPH_UNITS") : "6", nullptr, 10));
  const uint64_t seed0 = std::getenv("VGPU_JIT_RANDOM_SEED") ? std::strtoull(std::getenv("VGPU_JIT_RANDOM_SEED"), nullptr, 10) : 1;
  const std::string dir = tmp_dir("graphres");
  std::atomic<unsigned> next{0}, failures{0}, runs{0}, taken{0}, gen_ops{0};
  std::mutex mu;
  std::string first_failure;
  const auto worker = [&] {
    for (;;) {
      const unsigned unit = next.fetch_add(1);
      if (unit >= units) return;
      Gen g(seed0 * 9000011u + unit);
      jit::UnitSpec spec;
      spec.sm = 80;
      for (unsigned r = 0; r < 12; ++r) {
        spec.regions.push_back(graph_region(g, 8 + static_cast<unsigned>(g.u(50)), true, 10));
        for (const auto& o : spec.regions.back().ops) gen_ops += o.form ? 1u : 0u;
      }
      jit::Config on, off;
      on.cache_dir = dir + "/on" + std::to_string(unit);
      off.cache_dir = dir + "/off" + std::to_string(unit);
      off.residence = false;
      jit::Engine eon(on), eoff(off);
      const jit::Result ron = eon.obtain(spec), roff = eoff.obtain(spec);
      if (!ron.found || !roff.found) {
        std::lock_guard<std::mutex> lk(mu);
        ++failures;
        if (first_failure.empty()) first_failure = "unit " + std::to_string(unit) + ": compile failed: " + (ron.found ? roff.message : ron.message);
        continue;
      }
      std::mt19937_64 rng(seed0 * 15485863u + unit);
      for (unsigned r = 0; r < spec.regions.size(); ++r) {
        for (unsigned t = 0; t < 30; ++t) {
          GSim a;
          randomize_g(a, rng, false);
          a.seed = rng();
          const Trial tr = draw_trial(rng, spec.regions[r], t);
          if (t % 7 == 3) a.throw_at = static_cast<long>(rng() % 4);
          GSim b = a;
          jit::Ctx ca = make_gctx(a, tr.flags), cb = make_gctx(b, tr.flags);
          bool ta = false, tb = false;
          const uint32_t gl = static_cast<uint32_t>(__builtin_popcount(tr.group));
          try { ron.lib->fns[r](&ca, tr.group, gl, tr.entry, tr.n0); } catch (const std::runtime_error&) { ta = true; }
          try { roff.lib->fns[r](&cb, tr.group, gl, tr.entry, tr.n0); } catch (const std::runtime_error&) { tb = true; }
          a.exit_idx = ca.exit_idx; a.exit_ps = ca.exit_ps;
          b.exit_idx = cb.exit_idx; b.exit_ps = cb.exit_ps;
          ++runs;
          taken += a.extra > 0 ? 1u : 0u;
          if (ta != tb || !same_g(a, b, !ta)) {
            std::lock_guard<std::mutex> lk(mu);
            ++failures;
            if (first_failure.empty())
              first_failure = "unit " + std::to_string(unit) + " region " + std::to_string(r) + " trial " + std::to_string(t) + ": residence differs from the register-file code (R " +
                              (std::memcmp(a.R, b.R, sizeof a.R) ? "differs" : "same") + ", p " + (std::memcmp(a.p, b.p, sizeof a.p) ? "differs" : "same") + ", exit " +
                              std::to_string(a.exit_idx) + "/" + std::to_string(b.exit_idx) + ", log " + (a.log == b.log ? "same" : "differs") + ")";
          }
        }
      }
    }
  };
  const unsigned nthreads = std::min<unsigned>(units, std::max(1u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < nthreads; ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  std::printf("  graph + residence: %u units, %u runs, %u generated ops, %u with a branch or cont op taken, %u failures\n", units, runs.load(), gen_ops.load(), taken.load(),
              failures.load());
  std::error_code ec;
  fs::remove_all(dir, ec);
  if (failures) throw vtest::Failure(first_failure);
  VCHECK(taken.load() > runs.load() / 10);
}

VTEST_MAIN
