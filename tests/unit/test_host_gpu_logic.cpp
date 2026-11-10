// The host-GPU mode's decisions that need no card: which kernels may run on it with the simulator's bits, the chunk residency
// protocol, and the size rule (include/vgpu/host_gpu.hpp). The part that drives the real GPU is run by
// nvidia/tests/e2e/run_host_gpu.sh, on a machine with a card.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "vgpu/host_gpu.hpp"
#include "vtest.hpp"

using namespace vgpu::hostgpu;

VTEST(mode_is_exact_or_off) {
  VCHECK(mode_from_text("exact") == Mode::Exact);
  VCHECK(mode_from_text(nullptr) == Mode::Off);
  VCHECK(mode_from_text("") == Mode::Off);
  VCHECK(mode_from_text("approx") == Mode::Off);   // not built: a typo must not turn a mode on
  VCHECK(mode_from_text("EXACT") == Mode::Off);
  VCHECK(mode_from_text("1") == Mode::Off);
}

VTEST(exact_set_takes_integer_and_ieee_code) {
  for (const char* m : {"IMAD", "IADD3", "LOP3", "PLOP3", "LEA", "SHF", "ISETP", "MOV", "SEL", "PRMT", "FADD", "FMUL", "FFMA", "FSETP", "DFMA", "DADD", "DMUL",
                        "I2FP", "F2I", "LDG", "STG", "LDS", "STS", "S2R", "BAR", "BRA", "EXIT", "SHFL", "VOTE", "FCHK", "FRND"})
    VCHECK(exact_opcode_allowed(m, {}));
  VCHECK(exact_opcode_allowed("FADD", {"FTZ"}));
  VCHECK(exact_opcode_allowed("FFMA", {"RM"}));
  VCHECK(exact_opcode_allowed("RED", {"E", "ADD", "STRONG", "GPU"}));   // an integer reduction: the order of the adds is not visible
}

VTEST(exact_set_refuses_what_differed_in_the_spike) {
  // Every kernel of the sampled launches that differed from the card held one of these (09 doc, M2).
  for (const char* m : {"MUFU", "HMMA", "IMMA", "DMMA", "TEX", "TLD", "ATOMG", "ATOMS", "ATOM", "CALL", "RET", "HFMA2", "BMOV.UNKNOWN", "VPRINTF"})
    VCHECK(!exact_opcode_allowed(m, {}));
  VCHECK(!exact_opcode_allowed("RED", {"E", "ADD", "F32", "FTZ", "RN", "STRONG", "GPU"}));   // float reduction: the order changes the bits
  VCHECK(!exact_opcode_allowed("RED", {"E", "ADD", "F64"}));
  VCHECK(!exact_opcode_allowed("RED", {"E", "CAS"}));
  VCHECK(!exact_opcode_allowed("RED", {"E", "EXCH"}));
  VCHECK(!exact_opcode_allowed("", {}));
  VCHECK(!exact_opcode_allowed("imad", {}));   // names are SASS's, upper case; an unknown spelling is unknown
}

// The code of a few real kernels, as the opcode sets the trace of the spike showed: the verdict has to be the one the card gave.
static bool kernel_ok(const std::vector<std::pair<std::string, std::vector<std::string>>>& code) {
  for (const auto& [m, mods] : code)
    if (!exact_opcode_allowed(m, mods)) return false;
  return true;
}
VTEST(exact_set_matches_the_measured_kernels) {
  // PolyBench gemm_kernel: integer address arithmetic, FFMA, loads, stores: agreed.
  VCHECK(kernel_ok({{"S2R", {}}, {"IMAD", {}}, {"IADD3", {}}, {"LDG", {"E"}}, {"FFMA", {}}, {"FMUL", {}}, {"STG", {"E"}}, {"ISETP", {"GE", "AND"}}, {"BRA", {}}, {"EXIT", {}}}));
  // Rodinia lud_internal (shared memory, barrier): agreed.
  VCHECK(kernel_ok({{"LDS", {}}, {"STS", {}}, {"BAR", {"SYNC"}}, {"FFMA", {}}, {"LDG", {"E"}}, {"STG", {"E"}}}));
  // HeCBench silu forward (expf: MUFU.EX2 and MUFU.RCP): differed by one ulp.
  VCHECK(!kernel_ok({{"FFMA", {}}, {"MUFU", {"EX2"}}, {"MUFU", {"RCP"}}, {"FCHK", {}}, {"FADD", {"FTZ"}}}));
  // HeCBench aop compute_partial_beta (double atomicAdd as a compare-and-swap loop): card-nondeterministic.
  VCHECK(!kernel_ok({{"DADD", {}}, {"ATOMS", {"CAST", "SPIN", "64"}}, {"DFMA", {}}}));
  // HeCBench bh TreeBuildingKernel (ATOMG.E.CAS): card-nondeterministic.
  VCHECK(!kernel_ok({{"FSETP", {"GEU", "AND"}}, {"ATOMG", {"E", "CAS", "STRONG", "GPU"}}}));
}

// ---- residency: a reader on either side always sees the last write ------------------------------------------------------------

namespace {
struct World {
  uint64_t n;
  Residency res;
  std::vector<int> truth, host, card;
  int next = 1;
  explicit World(uint64_t chunks) : n(chunks), res(chunks), truth(chunks, 0), host(chunks, 0), card(chunks, -1) {}

  void sim_write(uint64_t c, bool whole) {
    for (uint64_t p : res.sim_write(c, c, whole)) host[p] = card[p];   // a partial write needs the chunk's bytes first
    if (!whole) VCHECK_EQ(host[c], truth[c]);                          // ... and the chunk is current before it is merged into
    host[c] = truth[c] = next++;
  }
  void sim_read(uint64_t c) {
    for (uint64_t p : res.sim_read(c, c)) host[p] = card[p];
    VCHECK_EQ(host[c], truth[c]);
  }
  void sim_launch(std::mt19937& rng) {   // the simulator runs a launch: Card chunks come back first, then it may write any chunk
    for (uint64_t p : res.sim_launch()) host[p] = card[p];
    for (uint64_t c = 0; c < n; ++c) VCHECK_EQ(host[c], truth[c]);
    for (uint64_t c = 0; c < n; ++c)
      if (rng() % 3 == 0) host[c] = truth[c] = next++;
    VCHECK_EQ(res.card_valid(), uint64_t{0});   // nothing of the card's copy is valid any more
  }
  void card_launch(std::mt19937& rng) {
    const std::vector<uint64_t> to_push = res.host_chunks();
    for (uint64_t p : res.card_prepare()) card[p] = host[p];
    VCHECK(res.host_chunks().empty());
    (void)to_push;
    for (uint64_t c = 0; c < n; ++c) VCHECK_EQ(card[c], truth[c]);     // the card sees every chunk as the simulator last left it
    for (uint64_t c = 0; c < n; ++c)
      if (rng() % 3 == 0) card[c] = truth[c] = next++;                 // the kernel writes some chunks
    res.card_launched();
  }
};
}  // namespace

VTEST(residency_keeps_both_sides_coherent) {
  for (uint32_t seed = 1; seed <= 200; ++seed) {
    std::mt19937 rng(seed);
    World w(1 + seed % 13);
    for (int step = 0; step < 400; ++step) {
      const uint64_t c = rng() % w.n;
      switch (rng() % 7) {
        case 0: w.sim_write(c, true); break;
        case 1: w.sim_write(c, false); break;
        case 2:
        case 3: w.sim_read(c); break;
        case 4: w.sim_launch(rng); break;
        default: w.card_launch(rng); break;
      }
    }
  }
}

// The counts are kept by every change of a chunk, not recounted; the gate's fast path trusts `access_changes_state` to skip only what is a no-op.
namespace {
void check_counts(const Residency& r) {
  uint64_t n[3] = {0, 0, 0};
  for (uint64_t c = 0; c < r.chunks(); ++c) ++n[static_cast<unsigned>(r.at(c))];
  VCHECK_EQ(r.count(Where::Host), n[0]);
  VCHECK_EQ(r.count(Where::Card), n[1]);
  VCHECK_EQ(r.count(Where::Both), n[2]);
  VCHECK_EQ(r.card_valid(), n[1] + n[2]);
  VCHECK_EQ(r.host_chunks().size(), n[0]);
}
bool same_states(const Residency& a, const Residency& b) {
  for (uint64_t c = 0; c < a.chunks(); ++c)
    if (a.at(c) != b.at(c)) return false;
  return true;
}
}  // namespace

VTEST(residency_counts_match_a_recount_after_every_operation) {
  for (uint32_t seed = 1; seed <= 300; ++seed) {
    std::mt19937 rng(seed);
    const uint64_t n = 1 + seed % 17;
    Residency r(n);
    check_counts(r);
    for (int step = 0; step < 300; ++step) {
      const uint64_t a = rng() % n, b = a + rng() % (n - a);
      switch (rng() % 9) {
        case 0: r.sim_write(a, b, rng() % 2); break;
        case 1: r.sim_read(a, b); break;
        case 2: r.card_prepare(); break;
        case 3: r.card_launched(); break;
        case 4: r.card_launched_writing(a, b); break;
        case 5: r.sim_launch(); break;
        case 6: r.set(a, static_cast<Where>(rng() % 3)); break;
        case 7: r.sim_read(a, b + 3); break;   // past the end: clamped
        default: r.sim_write(a, b + 3, true); break;
      }
      check_counts(r);
    }
  }
}

VTEST(access_changes_state_is_exactly_whether_the_gate_would_do_anything) {   // a skipped access must be one that the full path would leave alone
  for (uint32_t seed = 1; seed <= 300; ++seed) {
    std::mt19937 rng(seed);
    const uint64_t n = 1 + seed % 11;
    Residency r(n);
    for (int step = 0; step < 200; ++step) {
      switch (rng() % 4) {
        case 0: r.card_launched(); break;
        case 1: r.set(rng() % n, static_cast<Where>(rng() % 3)); break;
        case 2: r.card_prepare(); break;
        default: break;
      }
      const uint64_t a = rng() % n, b = a + rng() % (n - a);
      for (bool write : {false, true}) {
        Residency full = r;
        const std::vector<uint64_t> pulled = write ? full.sim_write(a, b, false) : full.sim_read(a, b);
        const bool changes = !pulled.empty() || !same_states(full, r);
        VCHECK_EQ(r.access_changes_state(a, b, write), changes);
      }
    }
  }
}

VTEST(residency_queries_do_not_grow_with_the_allocation) {   // the gate asked for the counts after every access: 2 x 1 M chunks scanned each time
  Residency r(1u << 20);   // 64 GiB of chunks
  r.card_launched();
  r.sim_read(0, 3);
  const auto t0 = std::chrono::steady_clock::now();
  uint64_t sink = 0;
  for (uint64_t i = 0; i < 200000; ++i) {
    sink += r.card_valid() + r.count(Where::Card) + (r.access_changes_state(i, i, false) ? 1 : 0);
    r.sim_read(i % 8, i % 8);
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  VCHECK(sink > 0);
  VCHECK(s < 2.0);   // microseconds in all; a scan of the chunks took more than 20 s for this many queries
}

VTEST(chunk_runs_group_adjacent_chunks) {
  VCHECK(chunk_runs({}).empty());
  const auto r = chunk_runs({0, 1, 2, 5, 7, 8});
  VCHECK_EQ(r.size(), size_t{3});
  VCHECK(r[0] == std::make_pair(uint64_t{0}, uint64_t{3}));
  VCHECK(r[1] == std::make_pair(uint64_t{5}, uint64_t{1}));
  VCHECK(r[2] == std::make_pair(uint64_t{7}, uint64_t{2}));
}

VTEST(chunk_span_knows_what_a_write_covers) {
  const uint64_t K = 64 * 1024, base = 0x2000'0000'0000ull;
  ChunkSpan s = chunk_span(base, 4 * K, base, 4 * K, K);   // the whole allocation
  VCHECK(s.any && s.first == 0 && s.last == 3 && s.first_whole && s.last_whole);
  s = chunk_span(base, 4 * K, base + 10, 100, K);          // a few bytes inside chunk 0
  VCHECK(s.any && s.first == 0 && s.last == 0 && !s.first_whole && !s.last_whole);
  s = chunk_span(base, 4 * K, base + K, K, K);             // exactly chunk 1
  VCHECK(s.first == 1 && s.last == 1 && s.first_whole && s.last_whole);
  s = chunk_span(base, 4 * K, base + K + 1, 2 * K, K);     // runs from inside chunk 1 into chunk 3
  VCHECK(s.first == 1 && s.last == 3 && !s.first_whole && !s.last_whole);
  s = chunk_span(base, 3 * K + 100, base + 3 * K, 100, K); // the short last chunk, covered in full
  VCHECK(s.first == 3 && s.last == 3 && s.first_whole && s.last_whole);
  s = chunk_span(base, 4 * K, base + 5 * K, 10, K);        // outside the allocation
  VCHECK(!s.any);
  s = chunk_span(base, 4 * K, base - 100, 200, K);         // starts below it: clamped
  VCHECK(s.any && s.first == 0 && s.last == 0 && !s.first_whole);
  s = chunk_span(base, 4 * K, base, 0, K);                 // empty
  VCHECK(!s.any);
}

VTEST(residency_moves_only_what_it_must) {
  Residency r(8);
  VCHECK_EQ(r.count(Where::Host), uint64_t{8});
  VCHECK_EQ(r.card_prepare().size(), size_t{8});   // everything is new: all pushed
  VCHECK_EQ(r.count(Where::Both), uint64_t{8});
  VCHECK_EQ(r.card_prepare().size(), size_t{0});   // a second launch with no simulator writes in between pushes nothing
  r.card_launched();
  VCHECK_EQ(r.count(Where::Card), uint64_t{8});
  VCHECK_EQ(r.sim_read(2, 3).size(), size_t{2});   // the simulator reads two chunks: two pulled
  VCHECK_EQ(r.sim_read(2, 3).size(), size_t{0});   // and they are Both: no second pull
  VCHECK_EQ(r.sim_write(2, 2, true).size(), size_t{0});   // a whole-chunk overwrite needs nothing from the card
  VCHECK(r.at(2) == Where::Host);
  VCHECK_EQ(r.sim_write(5, 5, false).size(), size_t{1});  // a partial write of a Card chunk pulls it first
  VCHECK_EQ(r.card_prepare().size(), size_t{2});          // chunks 2 and 5 are the simulator's again: pushed; 3 is Both, the others Card
  VCHECK(r.at(4) == Where::Card);                         // a launch that follows still treats a Card chunk as the card's
}

// ---- the size rule ------------------------------------------------------------------------------------------------------------

VTEST(size_rule_refuses_small_launches) {
  Costs c;
  Estimate e;
  e.sim_s = 50e-6;   // the simulator needs 50 us: nothing the card does can beat that by the margin
  VCHECK(!should_offload(e, c));
  e.sim_s = 150e-6;  // below the floor, whatever the card path costs
  VCHECK(!should_offload(e, c));
}

VTEST(size_rule_weighs_the_copies) {
  Costs c;
  Estimate e;
  e.sim_s = 0.9;                // a 900 ms simulated gemm (PolyBench) with 3 MiB of memory
  e.push_bytes = e.pull_bytes = 3u << 20;
  VCHECK(should_offload(e, c));
  e.sim_s = 0.01;               // 10 ms of simulation with a GiB to move each way: the copies cost more than the margin allows
  e.push_bytes = e.pull_bytes = 1ull << 30;
  VCHECK(!should_offload(e, c));
  e.sim_s = 0.5;                // the same data, a long kernel: 1 GiB is ~0.19 s both ways, 2x margin = 0.38 s < 0.5 s
  VCHECK(should_offload(e, c));
  e.sim_s = 0.3;
  VCHECK(!should_offload(e, c));
}

VTEST(size_rule_margin_is_applied_to_the_whole_path) {
  Costs c;
  c.margin = 2.0;
  Estimate e;
  e.push_bytes = 0;
  e.pull_bytes = 0;
  e.card_kernel_s = 1e-3;
  const double path = card_path_seconds(e, c);
  VCHECK(path > 1e-3 && path < 1.1e-3);
  e.sim_s = 2.0 * path * 0.99;
  VCHECK(!should_offload(e, c));
  e.sim_s = 2.0 * path * 1.01;
  VCHECK(should_offload(e, c));
}

VTEST(sim_time_table_needs_two_samples_and_keeps_the_smallest) {
  SimTimeTable t;
  VCHECK(!t.predict("k", 10).has_value());   // never seen: the first launch is the simulator's
  t.record("k", 10, 0.2);                    // a cold first launch (decode, set-up): 20 ms per block, under kOneSampleMinSeconds
  VCHECK(!t.predict("k", 10).has_value());   // one sample is not enough: it is the cold one
  t.record("k", 10, 0.05);                   // the warm launch: 5 ms per block
  VCHECK(t.predict("k", 10).has_value());
  VCHECK(*t.predict("k", 10) > 0.049 && *t.predict("k", 10) < 0.051);     // the smaller of the two
  VCHECK(*t.predict("k", 20) > 0.099 && *t.predict("k", 20) < 0.101);     // twice the blocks, twice the time
  VCHECK(!t.predict("other", 10).has_value());
  t.record("k", 10, 9.0);                    // a slower launch never raises the estimate: it errs towards the simulator
  VCHECK(*t.predict("k", 10) < 0.051);
  t.record("k", 10, 0.01);                   // a faster one lowers it
  VCHECK(*t.predict("k", 10) < 0.011);
}

VTEST(a_launch_that_lost_to_the_estimate_ends_offloading) {
  VCHECK(offload_paid_off(0.010, 0.050));
  VCHECK(offload_paid_off(0.050, 0.050));
  VCHECK(!offload_paid_off(0.100, 0.003));   // 3DCONV: 100 ms of copies for a 3 ms simulation
}

constexpr uint64_t kMiB = 1ull << 20;

VTEST(traffic_ban_constants_are_the_replayed_ones) {   // chosen on the 64 saved traces (docs/perf-design/rounds/08-rounds-sp11-sp15.md, Round sp14); changing one needs a new replay
  VCHECK(kTrafficLaunches == 2 && kTrafficSharePercent == 25 && kTrafficMinBytes == 12 * kMiB && kTrafficSlowShare == 0.5);
}

VTEST(traffic_ban_fires_on_the_second_heavy_launch_in_a_row) {   // LU's shape: the whole 16 MiB buffer pushed again every launch
  TrafficTable t;
  VCHECK(!t.record("lu", 16 * kMiB, 16 * kMiB, 0.004, 0.006));   // one heavy launch: not yet
  VCHECK(t.record("lu", 16 * kMiB, 16 * kMiB, 0.004, 0.006));    // two in a row: banned
  VCHECK(t.record("lu", 16 * kMiB, 16 * kMiB, 0.004, 0.006));    // and it stays so while it keeps pushing
}

VTEST(traffic_ban_needs_the_launches_in_a_row) {
  TrafficTable t;
  VCHECK(!t.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(!t.record("k", 0, 16 * kMiB, 0.0001, 0.006));            // a launch that pushed nothing resets the count
  VCHECK(!t.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(t.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
}

VTEST(traffic_ban_ignores_a_chain_without_pushes_and_the_first_upload) {
  TrafficTable t;
  VCHECK(!t.record("fdtd", 48 * kMiB, 48 * kMiB, 0.1, 0.1));      // the initial upload: heavy once (FDTD-2D)
  for (int i = 0; i < 100; ++i) VCHECK(!t.record("fdtd", 0, 48 * kMiB, 0.0002, 0.1));   // then nothing moves
  TrafficTable u;
  for (int i = 0; i < 100; ++i) VCHECK(!u.record("chain", 0, 48 * kMiB, 0.0002, 0.1));
}

VTEST(traffic_ban_spares_a_big_push_that_is_a_small_share) {   // base64e: 8 MiB of hundreds, and aligned-types: big buffers pushed in pieces
  TrafficTable t;
  for (int i = 0; i < 10; ++i) VCHECK(!t.record("b64", 100 * kMiB, 1000 * kMiB, 0.5, 0.6));   // 10% of live: under the share
  for (int i = 0; i < 10; ++i) VCHECK(!t.record("b64b", 249 * kMiB, 1000 * kMiB, 0.5, 0.6));  // 24.9%
  TrafficTable v;
  VCHECK(!v.record("exact", 250 * kMiB, 1000 * kMiB, 0.5, 0.6));   // exactly the share: not above it
  VCHECK(!v.record("exact", 250 * kMiB, 1000 * kMiB, 0.5, 0.6));
  VCHECK(!v.record("over", 251 * kMiB, 1000 * kMiB, 0.5, 0.6));
  VCHECK(v.record("over", 251 * kMiB, 1000 * kMiB, 0.5, 0.6));     // one MiB over: the same pair bans
}

VTEST(traffic_ban_spares_a_small_push) {   // a share of a small heap is not traffic worth a ban
  TrafficTable t;
  for (int i = 0; i < 10; ++i) VCHECK(!t.record("small", 4 * kMiB, 4 * kMiB, 0.004, 0.006));
  TrafficTable e;
  VCHECK(!e.record("edge", 12 * kMiB, 12 * kMiB, 0.004, 0.006));   // exactly the floor: not above it
  VCHECK(!e.record("edge", 12 * kMiB, 12 * kMiB, 0.004, 0.006));
  VCHECK(!e.record("above", 12 * kMiB + 1, 12 * kMiB + 1, 0.004, 0.006));
  VCHECK(e.record("above", 12 * kMiB + 1, 12 * kMiB + 1, 0.004, 0.006));
}

VTEST(traffic_ban_spares_a_launch_the_card_path_wins_by_far) {   // GRAMSCHM's shape: huge pushes, but the simulator would take 3x longer
  TrafficTable t;
  for (int i = 0; i < 10; ++i) VCHECK(!t.record("gs", 48 * kMiB, 48 * kMiB, 0.040, 0.120));   // 0.33 of the estimate
  TrafficTable u;
  VCHECK(!u.record("edge", 48 * kMiB, 48 * kMiB, 0.05, 0.10));   // exactly half the estimate: not past it
  VCHECK(!u.record("edge", 48 * kMiB, 48 * kMiB, 0.05, 0.10));
  VCHECK(!u.record("past", 48 * kMiB, 48 * kMiB, 0.0501, 0.10));
  VCHECK(u.record("past", 48 * kMiB, 48 * kMiB, 0.0501, 0.10));
}

VTEST(traffic_ban_is_per_kernel) {
  TrafficTable t;
  VCHECK(!t.record("a", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(!t.record("b", 16 * kMiB, 16 * kMiB, 0.004, 0.006));   // another kernel's first heavy launch is not a's second
  VCHECK(!t.record("c", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(t.record("a", 16 * kMiB, 16 * kMiB, 0.004, 0.006));    // a's second
  VCHECK(t.record("b", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  TrafficTable u;
  VCHECK(!u.record("a", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(!u.record("b", 0, 16 * kMiB, 0.0, 0.006));             // b's quiet launch does not reset a
  VCHECK(u.record("a", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
}

VTEST(traffic_ban_first_card_launch_counts_traffic_only) {   // its path includes the module load: no time verdict from it
  TrafficTable t;
  VCHECK(!t.record("k", 16 * kMiB, 16 * kMiB, 0.0001, 0.006, false));   // timing not valid: traffic alone makes it heavy
  VCHECK(t.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006, true));
  TrafficTable u;
  VCHECK(!u.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
  VCHECK(!u.record("k", 16 * kMiB, 16 * kMiB, 0.0001, 0.006, true));    // a valid, fast launch resets
  VCHECK(!u.record("k", 16 * kMiB, 16 * kMiB, 0.004, 0.006));
}

VTEST(traffic_ban_with_no_live_bytes_never_fires) {
  TrafficTable t;
  for (int i = 0; i < 5; ++i) VCHECK(!t.record("k", 16 * kMiB, 0, 0.004, 0.006));
}

VTEST(static_estimate_scales_with_the_grid) {   // an estimate, not a bound: see host_gpu.hpp
  Costs c;
  // PolyBench gemm_kernel: 1,024 blocks of 256 threads, about 60 static instructions: 5 ms of simulation by this count (it took 900 ms).
  const double a = static_sim_seconds(1024, 256, 60, c);
  VCHECK(a > 4.9e-3 && a < 5.0e-3);
  VCHECK(static_sim_seconds(2048, 256, 60, c) > 1.99 * a);
  VCHECK(static_sim_seconds(1024, 33, 60, c) > 1.99 * static_sim_seconds(1024, 32, 60, c));   // 33 threads is two warps
  Estimate e;
  e.sim_s = a;
  e.push_bytes = e.pull_bytes = 3u << 20;
  VCHECK(should_offload(e, c));       // a 5 ms lower bound already pays for 3 MiB each way
  e.sim_s = static_sim_seconds(2, 32, 60, c);   // a two-block launch: far below the floor
  VCHECK(!should_offload(e, c));
}

VTEST(back_edge_scan) {   // (pc, target) of every branch of a kernel
  VCHECK(!has_back_edge({}));
  VCHECK(!has_back_edge({{0x20, 0x80}, {0x40, 0x60}}));   // forward branches only: a bounds check
  VCHECK(has_back_edge({{0x20, 0x80}, {0x90, 0x30}}));    // a loop's closing branch
  VCHECK(has_back_edge({{0x40, 0x40}}));                  // a branch to itself: a spin
  VCHECK(has_back_edge({{0x40, 0x0}}));
}

// A PolyBench gemm_kernel launch on an A10G (80 multiprocessors): a loop, 1,024 blocks of 256 threads, about 60 static instructions, three 4 MiB matrices.
static FirstLaunch gemm_shape() {
  FirstLaunch f;
  f.blocks = 1024;
  f.threads_per_block = 256;
  f.static_instrs = 60;
  f.back_edge = true;
  f.multiprocessors = 80;
  f.live_bytes = 12ull << 20;
  return f;
}

VTEST(first_launch_takes_a_looping_kernel_on_a_full_grid) {
  Costs c;
  VCHECK(first_launch_offload(gemm_shape(), c));
}

VTEST(first_launch_refuses_a_kernel_without_a_loop) {   // 3DCONV: a stencil, threads leave at a bounds check; the static count was 38x off
  Costs c;
  FirstLaunch f = gemm_shape();
  f.back_edge = false;
  VCHECK(!first_launch_offload(f, c));
}

VTEST(first_launch_refuses_a_grid_that_does_not_fill_the_machine) {   // MVT: 16 blocks
  Costs c;
  FirstLaunch f = gemm_shape();
  f.static_instrs = 4000;   // so large that the margin is never what refuses: only the grid is under test
  VCHECK(first_launch_offload(f, c));
  f.blocks = 16;
  VCHECK(!first_launch_offload(f, c));
  f.blocks = 8 * f.multiprocessors - 1;
  VCHECK(!first_launch_offload(f, c));
  f.blocks = 8 * f.multiprocessors;   // exactly 8 per multiprocessor is enough
  VCHECK(first_launch_offload(f, c));
  f.multiprocessors = 0;              // a profile that does not say
  VCHECK(!first_launch_offload(f, c));
}

VTEST(first_launch_caps_what_a_mistaken_launch_may_move) {   // 3DCONV pushed 128 MiB per launch; the cap is 256 MiB of live memory
  Costs c;
  FirstLaunch f = gemm_shape();
  f.static_instrs = 100000;               // so large that the margin is no obstacle
  f.live_bytes = 256ull << 20;
  VCHECK(first_launch_offload(f, c));
  f.live_bytes = (256ull << 20) + 1;
  VCHECK(!first_launch_offload(f, c));
}

VTEST(first_launch_needs_the_estimate_to_beat_the_card_path) {
  Costs c;
  FirstLaunch f = gemm_shape();
  f.static_instrs = 2;                    // a few instructions per thread: the copies cost more than the simulator would
  VCHECK(!first_launch_offload(f, c));
  f.static_instrs = 60;
  f.live_bytes = 200ull << 20;            // the same kernel on 200 MiB: 33 ms of copies against 5 ms of simulation
  VCHECK(!first_launch_offload(f, c));
}

VTEST(sim_time_table_counts_samples) {
  SimTimeTable t;
  VCHECK(t.samples("k") == 0);
  t.record("k", 4, 0.1);
  VCHECK(t.samples("k") == 1 && !t.predict("k", 4));
  t.record("k", 4, 0.1);
  VCHECK(t.samples("k") == 2 && t.predict("k", 4));
  VCHECK(t.samples("other") == 0);
}

// ---- the one-sample rule: a first launch that took 0.25 s or more is estimate enough, for two offloaded launches ----
VTEST(one_sample_is_enough_for_a_long_first_launch) {
  SimTimeTable t;
  t.record("k", 100, kOneSampleMinSeconds);                 // exactly at the threshold: counts
  VCHECK(t.predict("k", 100).has_value());
  VCHECK(*t.predict("k", 100) > 0.249 && *t.predict("k", 100) < 0.251);
  VCHECK(*t.predict("k", 200) > 0.499 && *t.predict("k", 200) < 0.501);   // scaled by the grid
  SimTimeTable s;
  s.record("short", 100, kOneSampleMinSeconds * 0.99);      // just under: not a long launch
  VCHECK(!s.predict("short", 100).has_value());
}

VTEST(one_sample_allowance_ends_after_two_offloaded_launches) {
  SimTimeTable t;
  t.record("k", 10, 1.0);
  for (unsigned i = 0; i < kOneSampleOffloads; ++i) {
    VCHECK(t.predict("k", 10).has_value());                 // still within the allowance
    t.note_offload("k");
  }
  VCHECK(!t.predict("k", 10).has_value());                  // spent: the next launch is the simulator's, for the second sample
  t.record("k", 10, 0.4);                                   // the warm sample
  VCHECK(t.predict("k", 10).has_value());                   // two samples now: the usual rule, no allowance needed
  VCHECK(*t.predict("k", 10) > 0.39 && *t.predict("k", 10) < 0.41);
  t.note_offload("k");                                      // (offloads on two samples do not count against anything)
  VCHECK(t.predict("k", 10).has_value());
}

VTEST(one_sample_allowance_is_per_kernel) {
  SimTimeTable t;
  t.record("a", 10, 1.0);
  t.record("b", 10, 1.0);
  for (unsigned i = 0; i < kOneSampleOffloads; ++i) t.note_offload("a");
  VCHECK(!t.predict("a", 10).has_value());
  VCHECK(t.predict("b", 10).has_value());
  t.note_offload("never_seen");                             // no entry is made by an offload alone
  VCHECK(!t.predict("never_seen", 10).has_value());
  VCHECK(t.samples("never_seen") == 0);
}

// ---- the driver start is charged by the offload decision (sp43; docs/performance.md "Round sp43") ----------------------------------------------
// An estimate that passes the plain rule by a wide margin: 50 ms of simulation, 1 MiB each way.
static Estimate fifty_ms() {
  Estimate e;
  e.sim_s = 0.05;
  e.push_bytes = e.pull_bytes = 1u << 20;
  return e;
}

VTEST(the_start_cost_is_the_measured_one) {   // 0.23 s on an A10G (Round sp42), rounded up; a change needs a new measurement
  Costs c;
  VCHECK(c.start_cost_s == 0.25);
}

VTEST(a_launch_that_cannot_repay_the_start_does_not_start_the_driver) {   // Rodinia: 5 to 50 ms of simulation in the whole program, 0.04x to 0.29x with the start
  Costs c;
  VCHECK(should_offload(fifty_ms(), c));                                   // worth a card with the driver up ...
  VCHECK(!should_offload_with_start(fifty_ms(), c, false, 0.0));           // ... not before it is paid for
  VCHECK(should_offload_with_start(fifty_ms(), c, true, 0.0));             // and plainly so once it is up
}

VTEST(one_large_launch_repays_the_start_at_once) {   // PolyBench SYRK: 4 s of simulation in one launch
  Costs c;
  Estimate e;
  e.sim_s = 4.0;
  e.push_bytes = e.pull_bytes = 8u << 20;
  VCHECK(should_offload_with_start(e, c, false, 0.0));
  e.sim_s = 0.4;   // (the margin: 0.4 s is less than twice the 0.25 s start plus the copies)
  VCHECK(!should_offload_with_start(e, c, false, 0.0));
  e.sim_s = 0.6;
  VCHECK(should_offload_with_start(e, c, false, 0.0));
}

VTEST(the_start_is_bought_when_as_much_was_spent_renting) {   // ski rental: a thousand launches of 50 ms, the driver starts after 0.25 s of them
  Costs c;
  VCHECK(!should_offload_with_start(fifty_ms(), c, false, 0.0));
  VCHECK(!should_offload_with_start(fifty_ms(), c, false, c.start_cost_s * 0.99));
  VCHECK(should_offload_with_start(fifty_ms(), c, false, c.start_cost_s));
  VCHECK(should_offload_with_start(fifty_ms(), c, false, 10.0));
}

VTEST(spent_time_never_sends_a_launch_the_plain_rule_refuses) {   // a tiny launch of a program that spent minutes elsewhere stays in the simulator
  Costs c;
  Estimate e;
  e.sim_s = 150e-6;   // under the floor
  VCHECK(!should_offload_with_start(e, c, false, 1000.0));
  VCHECK(!should_offload_with_start(e, c, true, 1000.0));
  e.sim_s = 0.01;     // 10 ms against a GiB to move each way
  e.push_bytes = e.pull_bytes = 1ull << 30;
  VCHECK(!should_offload_with_start(e, c, false, 1000.0));
}

VTEST(a_start_cost_of_zero_is_the_rule_before_sp43) {   // VGPU_HOST_GPU_START_COST=0
  Costs c;
  c.start_cost_s = 0;
  for (double s : {50e-6, 0.001, 0.05, 4.0}) {
    Estimate e = fifty_ms();
    e.sim_s = s;
    VCHECK_EQ(should_offload_with_start(e, c, false, 0.0), should_offload(e, c));
  }
}

VTEST(with_the_driver_up_the_decision_is_the_plain_rule) {   // for any account of spent time
  Costs c;
  for (double s : {50e-6, 0.001, 0.05, 4.0})
    for (double spent : {0.0, 0.1, 5.0}) {
      Estimate e = fifty_ms();
      e.sim_s = s;
      VCHECK_EQ(should_offload_with_start(e, c, true, spent), should_offload(e, c));
    }
}

VTEST(the_start_is_part_of_the_card_path) {
  Costs c;
  Estimate e = fifty_ms();
  const double warm = card_path_seconds(e, c);
  e.start_s = 0.25;
  VCHECK(card_path_seconds(e, c) > warm + 0.2499 && card_path_seconds(e, c) < warm + 0.2501);
}

VTEST(the_first_launch_rule_pays_the_start_too) {   // the static estimate of the gemm shape is 5 ms
  Costs c;
  VCHECK(first_launch_offload(gemm_shape(), c));                    // (default: the driver is up)
  VCHECK(!first_launch_offload(gemm_shape(), c, false, 0.0));       // a first launch of 5 ms estimated cannot start a driver
  FirstLaunch f = gemm_shape();
  f.static_instrs = 60000;                                          // 1000x the code: 5 s estimated
  VCHECK(first_launch_offload(f, c, false, 0.0));                   // that launch repays it
  VCHECK(first_launch_offload(gemm_shape(), c, false, c.start_cost_s));   // or the program has spent as much in the simulator already
}

// ---- the division idiom (host_gpu.hpp division_idiom_only), on the shape ptxas 13.0 gave `a / b` for sm_86 (cuobjdump of an own kernel) ----
using K = CodeInsn::Kind;
static CodeInsn ci(K kind, uint64_t pc, int64_t target = -1, int src0 = -1, int src1 = -1) {
  CodeInsn c;
  c.kind = kind;
  c.pc = pc;
  c.target = target;
  c.src0 = src0;
  c.src1 = src1;
  return c;
}
// main line: ... MUFU.RCP R8, R3 ; FCHK P0, R0, R3 ; FFMA x5 ; @!P0 BRA ; MOV ; CALL -> slow ; ... EXIT ; slow: ALU..., MUFU.RCP R8, R3, ALU..., RET
static std::vector<CodeInsn> division_kernel() {
  std::vector<CodeInsn> c;
  uint64_t pc = 0;
  const auto add = [&](K k, int64_t target = -1, int s0 = -1, int s1 = -1) { c.push_back(ci(k, pc, target, s0, s1)); pc += 16; };
  add(K::Other);                    // LDG is Memory only inside a subroutine: here it is the kernel's own load
  add(K::Other);
  add(K::MufuRcp, -1, 3);           // MUFU.RCP R8, R3
  add(K::Fchk, -1, 0, 3);           // FCHK P0, R0, R3
  for (int i = 0; i < 5; ++i) add(K::Other);   // FFMAs
  add(K::Other);                    // @!P0 BRA
  add(K::Other);                    // MOV R4, return
  add(K::Call, 0);                  // target patched below
  add(K::Other);
  add(K::Other);                    // STG
  add(K::Other);                    // EXIT
  const size_t slow_at = c.size();
  const uint64_t slow_pc = pc;
  add(K::Other);
  add(K::Other);
  add(K::MufuRcp, -1, 3);           // the slow path's own MUFU.RCP, no FCHK
  add(K::Other);
  add(K::Ret);
  for (CodeInsn& x : c)
    if (x.kind == K::Call) x.target = static_cast<int64_t>(slow_pc);
  (void)slow_at;
  return c;
}

VTEST(division_idiom_is_recognized) {
  VCHECK(division_idiom_only(division_kernel()));
}

VTEST(division_idiom_refuses_a_bare_reciprocal) {
  // rcp.approx.ftz.f32: one MUFU.RCP and nothing that makes the result independent of its last bit
  VCHECK(!division_idiom_only({ci(K::Other, 0), ci(K::MufuRcp, 16, -1, 3), ci(K::Other, 32), ci(K::Other, 48)}));
  // an FCHK of another register is not the divisor's
  auto c = division_kernel();
  for (CodeInsn& x : c)
    if (x.kind == K::Fchk) x.src1 = 4;
  VCHECK(!division_idiom_only(c));
  // an FCHK too far after the MUFU
  c = division_kernel();
  for (size_t i = 0; i < c.size(); ++i)
    if (c[i].kind == K::MufuRcp) {
      std::vector<CodeInsn> d(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(i + 1));
      for (unsigned k = 0; k < kIdiomMufuToFchk; ++k) d.push_back(ci(K::Other, 1000 + k));
      d.push_back(ci(K::Fchk, 2000, -1, 0, 3));
      VCHECK(!division_idiom_only(d));
      break;
    }
}

VTEST(division_idiom_refuses_other_sfu_and_calls) {
  auto c = division_kernel();
  c[0].kind = K::MufuOther;                        // an EX2 anywhere
  VCHECK(!division_idiom_only(c));
  c = division_kernel();
  for (CodeInsn& x : c)
    if (x.kind == K::Fchk) x.kind = K::Other;       // no FCHK: the CALL is not a division's slow path
  VCHECK(!division_idiom_only(c));
  c = division_kernel();
  c.back() = ci(K::Other, c.back().pc);             // a subroutine that never returns
  VCHECK(!division_idiom_only(c));
  c = division_kernel();
  c[c.size() - 2].kind = K::Memory;                 // a subroutine that touches memory is not the slow path
  VCHECK(!division_idiom_only(c));
  c = division_kernel();
  c.insert(c.begin() + 1, ci(K::Ret, 8));           // a RET outside any subroutine
  VCHECK(!division_idiom_only(c));
  c = division_kernel();
  for (CodeInsn& x : c)
    if (x.kind == K::Call) x.target = 9999;         // a call to an address that is not an instruction
  VCHECK(!division_idiom_only(c));
}

VTEST(division_idiom_refuses_a_device_function_call) {
  // a __noinline__ function: a CALL to a subroutine and a RET, with no division and no FCHK anywhere -- not the slow path, so not the idiom
  VCHECK(!division_idiom_only({ci(K::Other, 0), ci(K::Call, 16, 64), ci(K::Other, 32), ci(K::Other, 48), ci(K::Other, 64), ci(K::Ret, 80)}));
  // with an FCHK far before the call (outside kIdiomFchkToCall instructions) it is still not
  std::vector<CodeInsn> c = {ci(K::Fchk, 0, -1, 0, 1)};
  for (unsigned k = 1; k <= kIdiomFchkToCall + 1; ++k) c.push_back(ci(K::Other, 16 * k));
  const uint64_t call_pc = 16 * (kIdiomFchkToCall + 2);
  c.push_back(ci(K::Call, call_pc, static_cast<int64_t>(call_pc + 32)));
  c.push_back(ci(K::Other, call_pc + 16));
  c.push_back(ci(K::Other, call_pc + 32));
  c.push_back(ci(K::Ret, call_pc + 48));
  VCHECK(!division_idiom_only(c));
}

VTEST(division_idiom_takes_the_nan_generator_only_in_the_slow_path) {
  auto c = division_kernel();
  c[c.size() - 2].kind = K::NanGen;                 // inside the subroutine: the slow path's 0/0 case
  VCHECK(division_idiom_only(c));
  c = division_kernel();
  c[0].kind = K::NanGen;                            // in the kernel's own code: a bare NaN constant through an SFU instruction
  VCHECK(!division_idiom_only(c));
}

VTEST(division_idiom_takes_two_divisions_sharing_a_slow_path) {
  auto one = division_kernel();
  // a second MUFU.RCP/FCHK/CALL group before the slow path, calling the same subroutine
  std::vector<CodeInsn> c;
  uint64_t slow_pc = 0;
  for (const CodeInsn& x : one)
    if (x.kind == K::Call) slow_pc = static_cast<uint64_t>(x.target);
  for (const CodeInsn& x : one) {
    if (x.pc == slow_pc) {
      c.push_back(ci(K::MufuRcp, slow_pc - 1000, -1, 6));
      c.push_back(ci(K::Fchk, slow_pc - 999, -1, 5, 6));
      c.push_back(ci(K::Call, slow_pc - 998, static_cast<int64_t>(slow_pc)));
    }
    c.push_back(x);
  }
  VCHECK(division_idiom_only(c));
}

VTEST(idiom_classification_by_mnemonic) {
  VCHECK(classify_for_idiom("MUFU", {"RCP"}) == K::MufuRcp);
  VCHECK(classify_for_idiom("MUFU", {"RCP64H"}) == K::MufuOther);
  VCHECK(classify_for_idiom("MUFU", {"EX2"}) == K::MufuOther);
  VCHECK(classify_for_idiom("MUFU", {"RCP", "F16"}) == K::MufuOther);
  VCHECK(classify_for_idiom("MUFU", {"RSQ"}) == K::MufuOther);                  // an RSQ of a register is an SFU result
  VCHECK(classify_for_idiom("MUFU", {"RSQ"}, true) == K::NanGen);                // of an immediate NaN it is the division's NaN generator
  VCHECK(classify_for_idiom("MUFU", {"EX2"}, true) == K::MufuOther);
  VCHECK(classify_for_idiom("MUFU", {"RCP"}, true) == K::MufuRcp);
  VCHECK(classify_for_idiom("FCHK", {}) == K::Fchk);
  VCHECK(classify_for_idiom("CALL", {"REL", "NOINC"}) == K::Call);
  VCHECK(classify_for_idiom("RET", {"REL", "NODEC"}) == K::Ret);
  VCHECK(classify_for_idiom("LDG", {"E"}) == K::Memory);
  VCHECK(classify_for_idiom("SHFL", {}) == K::Memory);
  VCHECK(classify_for_idiom("HMMA", {}) == K::Memory);        // not in the exact set: not arithmetic of the slow path
  VCHECK(classify_for_idiom("FFMA", {"RM"}) == K::Other);
  VCHECK(classify_for_idiom("SHF", {"R", "U32", "HI"}) == K::Other);
  // the default exact set itself is unchanged by all of this
  VCHECK(!exact_opcode_allowed("MUFU", {"RCP"}));
  VCHECK(!exact_opcode_allowed("CALL", {"REL", "NOINC"}));
  VCHECK(!exact_opcode_allowed("RET", {"REL", "NODEC"}));
}

// The start-up refusals (08 section 3): a mode value that is not a mode is said, and every diagnostic that only the simulator gives switches the mode off.
VTEST(mode_text_known_and_diagnostic_switches) {
  VCHECK(mode_text_is_known(nullptr) && mode_text_is_known("") && mode_text_is_known("off") && mode_text_is_known("exact") && mode_text_is_known("0"));
  VCHECK(!mode_text_is_known("approx") && !mode_text_is_known("EXACT") && !mode_text_is_known("on") && !mode_text_is_known("1"));
  for (const char* v : kIncompatibleVars) {
    const std::string name(v);
    VCHECK(!diagnostic_switched_on(name, nullptr));
    VCHECK(!diagnostic_switched_on(name, ""));
    VCHECK(!diagnostic_switched_on(name, "0"));
    if (name != "VGPU_SCHEDULER") VCHECK(diagnostic_switched_on(name, "1"));
  }
  // the scheduler's default name is what a config file writes for `scheduler: deterministic`: not a diagnostic; the others are
  VCHECK(!diagnostic_switched_on("VGPU_SCHEDULER", "deterministic"));
  VCHECK(diagnostic_switched_on("VGPU_SCHEDULER", "random") && diagnostic_switched_on("VGPU_SCHEDULER", "adversarial"));
  // every way the simulator can be told to fail or look inside a launch is on the list
  const char* must[] = {"VGPU_COUNTERS", "VGPU_SASS_TRACE", "VGPU_KERNEL_DIGEST", "VGPU_TRACE", "VGPU_MAX_STEPS", "VGPU_FAIL_ALLOC", "VGPU_FAIL_LAUNCH",
                        "VGPU_FAIL_MEMCPY", "VGPU_RACE", "VGPU_STRICT", "VGPU_SCHEDULER"};
  for (const char* m : must) {
    bool found = false;
    for (const char* v : kIncompatibleVars) found = found || std::string(v) == m;
    if (!found) std::fprintf(stderr, "%s is not in kIncompatibleVars\n", m);
    VCHECK(found);
  }
}

VTEST_MAIN
