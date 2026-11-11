// Host-GPU mode, the logic that needs no card (docs/perf-design/09-host-gpu-spike.md).
//
// With `VGPU_HOST_GPU=exact` a launch may run on the real GPU of the machine instead of in the simulator, when its result is
// the same bits either way. Three decisions are made here, with no CUDA and no simulator state, so they are unit tested on any
// machine (tests/unit/test_host_gpu_logic.cpp):
//   * exact_opcode_allowed: may a kernel that holds this SASS instruction run on a card with the simulator's bits?
//   * Residency: which 64 KiB chunks of an allocation the simulator holds, the card holds, or both hold (the design for a
//     persistent card copy; the first prototype moves everything on every launch and does not need it, see the doc).
//   * should_offload / SimTimeTable: does the card path (launch, copy in, copy out) beat the simulator's estimate?
// The part that talks to the driver is src/runtime/host_gpu.cpp.
#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <utility>
#include <string>
#include <vector>

namespace vgpu::hostgpu {

enum class Mode { Off, Exact, Relaxed };

// "exact" and "relaxed" turn the mode on; anything else, empty or null is off. (`approx` is not a mode: a typo must not turn one on.)
// Relaxed (docs/perf-design/12-host-gpu-relaxed.md) also sends kernels to the card whose results are the card's own and not the simulator's bits.
inline Mode mode_from_text(const char* text) {
  if (!text) return Mode::Off;
  const std::string t(text);
  return t == "exact" ? Mode::Exact : t == "relaxed" ? Mode::Relaxed : Mode::Off;
}

// A mode value that is none of off, exact and relaxed (a typo, or `approx`, which is not a mode) must be said, never read as off in silence.
inline bool mode_text_is_known(const char* text) {
  if (!text || !*text) return true;
  const std::string t(text);
  return t == "off" || t == "exact" || t == "relaxed" || t == "0";
}

// The switches of diagnostics that only the simulator can give: with any of them on the mode refuses to start (08 section 3), never quietly drops them.
inline constexpr const char* kIncompatibleVars[] = {
    "VGPU_COUNTERS",   "VGPU_COUNTERS_FILE", "VGPU_SASS_TRACE", "VGPU_KERNEL_DIGEST", "VGPU_KERNEL_DIGEST_DUMP", "VGPU_HOSTGPU_CAPTURE", "VGPU_TRACE",
    "VGPU_MAX_STEPS",  "VGPU_FAIL_ALLOC",    "VGPU_FAIL_LAUNCH", "VGPU_FAIL_MEMCPY", "VGPU_RACE", "VGPU_STRICT", "VGPU_SCHEDULER", "VGPU_LOCKSTEP",
};

// Whether the value of one of kIncompatibleVars switches its diagnostic on. Unset, empty and 0 are off; the scheduler's default name is not a diagnostic
// (a config file that says `scheduler: deterministic` writes it).
inline bool diagnostic_switched_on(const std::string& name, const char* value) {
  if (!value || !*value) return false;
  const std::string v(value);
  if (v == "0") return false;
  if (name == "VGPU_SCHEDULER") return v != "deterministic";
  return true;
}

// ---- eligibility ------------------------------------------------------------------------------------------------------------

// The instructions whose result is defined bit for bit by the architecture and that the simulator computes as the card does:
// integer and IEEE float arithmetic and conversions, predicates, moves, loads and stores, barriers and plain control flow.
// Measured in the spike (09 doc, M2): every kernel of the Rodinia, PolyBench and HeCBench launches sampled whose code stayed
// inside this set agreed with the card to the bit, and every kernel that differed held one instruction outside it (MUFU, a
// compare-and-swap, a floating-point atomic). A mnemonic that is not named here is refused: an unknown instruction is not
// known to be exact.
inline bool exact_mnemonic_allowed(const std::string& m) {
  static const char* const kAllowed[] = {
      // integer
      "IMAD", "IADD3", "IADD", "LOP3", "LOP", "LEA", "SHF", "SHL", "SHR", "ISETP", "IMNMX", "IABS", "POPC", "FLO", "BREV", "PRMT",
      "SEL", "MOV", "R2P", "P2R", "PLOP3", "UPLOP3", "PSETP", "ISET", "FSET", "DSET", "I2I", "I2IP", "SGXT", "BMSK", "VIADD", "VIMNMX",
      // IEEE float and double (no SFU)
      "FADD", "FMUL", "FFMA", "FSETP", "FSEL", "FMNMX", "DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "I2F", "I2FP", "F2I", "F2F", "F2FP",
      "FRND", "FCHK",
      // memory (plain loads and stores; integer reductions below)
      "LDG", "STG", "LDS", "STS", "LDL", "STL", "LD", "ST", "LDC", "ULDC",
      // special registers, uniform datapath
      "S2R", "S2UR", "CS2R", "UMOV", "UIADD3", "UIMAD", "ULOP3", "USHF", "ULEA", "UISETP", "USEL", "UPRMT", "UFLO", "UPOPC", "UP2UR",
      "R2UR", "VOTEU",
      // control, barriers, warp
      "BRA", "BRX", "BSSY", "BSYNC", "BREAK", "BMOV", "EXIT", "NOP", "YIELD", "WARPSYNC", "BAR", "MEMBAR", "ERRBAR", "DEPBAR", "CCTL",
      "SHFL", "VOTE", "MATCH",
      // integer reductions with no returned value (order of the adds does not matter)
      "RED",
  };
  for (const char* a : kAllowed)
    if (m == a) return true;
  return false;
}

// Modifiers that turn an allowed mnemonic into an order- or implementation-dependent one: a floating-point reduction adds in
// whatever order the blocks arrive, and a reduction on a float is not the integer one.
inline bool exact_mods_allowed(const std::string& mnemonic, const std::vector<std::string>& mods) {
  if (mnemonic == "RED") {
    for (const std::string& m : mods)
      if (m == "F16" || m == "F32" || m == "F64" || m == "BF16" || m == "FTZ" || m == "CAS" || m == "EXCH" || m == "INC" || m == "DEC" || m == "SPIN" ||
          m == "CAST" || m == "POPC")
        return false;
  }
  return true;
}

inline bool exact_opcode_allowed(const std::string& mnemonic, const std::vector<std::string>& mods) {
  return exact_mnemonic_allowed(mnemonic) && exact_mods_allowed(mnemonic, mods);
}

// ---- the IEEE division idiom (MUFU.RCP + FCHK + a call to the slow path) --------------------------------------------------------------
// MUFU.RCP alone is NOT in the exact set: the card's reciprocal differs from the simulator's correctly rounded one by one ulp on about 13% of the mantissas
// (tools/probes/mufu_rcp_probe.cu, round 5), and 09 doc section 2 saw 1 to 171 ulp results downstream of an SFU instruction. But ptxas builds the IEEE
// quotient a / b (div.rn.f32, also `/` in CUDA C++) from MUFU.RCP, a few FFMAs that refine it, an FCHK that sends the rare operands to a slow path
// (CALL.REL.NOINC to a subroutine that also uses MUFU.RCP, ending in RET), and the result is the correctly rounded quotient whatever the last bit of the
// approximation: the PTX ISA requires div.rn.f32 to be IEEE-compliant, so the card and the simulator both give the one correctly rounded answer.
// Tested here without a card: the refinement over every mantissa of the divisor with the simulator's reciprocal and with it one ulp off either way
// (tests/unit/test_host_gpu_ref.cpp); on a card, by tools/probes/hostgpu_sfu_exhaustive.cu div: 12 dividends against all 2^32 divisors, 0 different on an A10G
// (docs/perf-design/rounds/11-rounds-sp36-sp41.md "Round sp44", B; other cards were not run). The recognizer is still behind VGPU_HOST_GPU_DIV=1 and the
// default exact set is unchanged.
//
// What the recognizer accepts (division_idiom_only): every MUFU is a plain RCP; each one outside a subroutine has an FCHK of the same divisor register within
// kIdiomMufuToFchk instructions after it; every CALL goes to a subroutine (from its target to the first RET at or after it) and has an FCHK within
// kIdiomFchkToCall instructions before it; a subroutine holds arithmetic only (no memory, barrier or warp operation, no call); every RET is in a subroutine.
// The slow path makes the NaN of 0/0 and inf/inf with `MUFU.RSQ R, -QNAN` (the reciprocal square root of an immediate NaN: the canonical NaN 0x7fffffff, which
// the simulator gives and the card run checks with dividends 0, inf and NaN): that one form (NanGen) is accepted inside a subroutine and nowhere else.
// A kernel with a bare rcp.approx (no FCHK), a device function that is not the slow path, or any other MUFU fails the check and stays refused.
struct CodeInsn {
  enum class Kind : uint8_t { Other, MufuRcp, MufuOther, Fchk, Call, Ret, Memory, NanGen };
  Kind kind = Kind::Other;
  uint64_t pc = 0;
  int64_t target = -1;    // CALL: the address of the subroutine
  int src0 = -1;          // MUFU: the register read; FCHK: the dividend's
  int src1 = -1;          // FCHK: the divisor's register
};
inline constexpr unsigned kIdiomMufuToFchk = 32;   // (the scheduler interleaves other work: 9 instructions apart in a kernel that also has an expf)
inline constexpr unsigned kIdiomFchkToCall = 12;

inline bool division_idiom_only(const std::vector<CodeInsn>& code, const char** why = nullptr) {
  const char* sink = "";
  if (!why) why = &sink;
  using K = CodeInsn::Kind;
  struct Region { uint64_t lo, hi; };
  std::vector<Region> regions;
  bool any = false;
  for (size_t i = 0; i < code.size(); ++i) {
    if (code[i].kind != K::Call) continue;
    any = true;
    // the subroutine: from the target to the first RET at or after it
    size_t t = code.size();
    for (size_t j = 0; j < code.size(); ++j)
      if (static_cast<int64_t>(code[j].pc) == code[i].target) { t = j; break; }
    if (t == code.size() || t <= i) { *why = "a call to an address that is not later in the code"; return false; }   // no such address, or a call backwards (a loop of calls is not the idiom)
    size_t r = t;
    while (r < code.size() && code[r].kind != K::Ret) {
      if (code[r].kind == K::Call || code[r].kind == K::Memory) { *why = "the subroutine calls or touches memory or other threads"; return false; }
      ++r;
    }
    if (r == code.size()) { *why = "a subroutine with no RET"; return false; }
    bool fchk_before = false;
    for (size_t j = i > kIdiomFchkToCall ? i - kIdiomFchkToCall : 0; j < i; ++j) fchk_before = fchk_before || code[j].kind == K::Fchk;
    if (!fchk_before) { *why = "a CALL with no FCHK before it"; return false; }
    regions.push_back({code[t].pc, code[r].pc});
  }
  const auto in_region = [&](uint64_t pc) {
    for (const Region& g : regions)
      if (pc >= g.lo && pc <= g.hi) return true;
    return false;
  };
  for (size_t i = 0; i < code.size(); ++i) {
    switch (code[i].kind) {
      case K::MufuOther: *why = "an SFU instruction other than RCP"; return false;
      case K::NanGen:
        if (!in_region(code[i].pc)) { *why = "a NaN generator outside a division subroutine"; return false; }
        break;
      case K::Ret:
        if (!in_region(code[i].pc)) { *why = "a RET outside a division subroutine"; return false; }
        break;
      case K::MufuRcp: {
        if (in_region(code[i].pc)) break;
        bool checked = false;
        for (size_t j = i + 1; j < code.size() && j <= i + kIdiomMufuToFchk && !checked; ++j)
          checked = code[j].kind == K::Fchk && code[j].src1 >= 0 && code[j].src1 == code[i].src0;
        if (!checked) { *why = "a MUFU.RCP with no FCHK of its divisor right after"; return false; }
        break;
      }
      default: break;
    }
  }
  (void)any;
  return true;
}

// The kind of an instruction for the check above, from its mnemonic and modifiers. Anything that touches memory, a barrier or other threads is Memory
// (not allowed inside the slow path); an instruction that is not in the exact set at all is Memory too, so a subroutine holding one fails.
inline CodeInsn::Kind classify_for_idiom(const std::string& mnemonic, const std::vector<std::string>& mods, bool source_is_nan_constant = false) {
  using K = CodeInsn::Kind;
  if (mnemonic == "MUFU") {
    bool rcp = false, rsq = false, other = false;
    for (const std::string& m : mods) (m == "RCP" ? rcp : m == "RSQ" ? rsq : other) = true;
    if (rsq && !rcp && !other && source_is_nan_constant) return K::NanGen;
    return rcp && !rsq && !other ? K::MufuRcp : K::MufuOther;
  }
  if (mnemonic == "FCHK") return K::Fchk;
  if (mnemonic == "CALL") return K::Call;
  if (mnemonic == "RET") return K::Ret;
  static const char* const kTouchesOthers[] = {"LDG", "STG", "LDS", "STS", "LDL", "STL", "LD", "ST", "BAR", "MEMBAR", "ERRBAR", "DEPBAR", "CCTL", "SHFL",
                                               "VOTE", "MATCH", "RED", "WARPSYNC", "VOTEU"};
  for (const char* t : kTouchesOthers)
    if (mnemonic == t) return K::Memory;
  return exact_mnemonic_allowed(mnemonic) ? K::Other : K::Memory;
}

// ---- relaxed mode: what a kernel's code is made of (docs/perf-design/12-host-gpu-relaxed.md) ------------------------------------------------
// VGPU_HOST_GPU=relaxed also sends kernels to the card whose result is the card's and not the simulator's bits. Every instruction of a kernel is put in
// one class, once, and the mode admits a kernel by the classes it holds. It stays a whitelist: a mnemonic that is not named below is Never.
enum class InsnClass : uint8_t {
  Exact,       // today's exact set
  Mufu,        // every MUFU form (the card's approximation is 1 to 2 ulp from the simulator's, sp44)
  Half,        // packed half arithmetic: HADD2, HMUL2, HFMA2, HSETP2, HSET2, HMNMX2
  FloatRed,    // a float reduction with no returned value: the sum depends on the order the threads arrive in
  Call,        // CALL and RET (the runtime also checks that every call stays inside the kernel's own code section)
  Extra,       // plain compute and memory instructions the exact set never measured: they read and write registers, shared memory and the same global
               // addresses as LDG and STG, so the card needs nothing that only the simulator holds
  Tensor,      // HMMA, IMMA, DMMA
  AtomInt,     // integer atomics that return the old value (ADD, MIN, MAX, AND, OR, XOR, INC, DEC): memory ends the same, the returned values depend on the order
  AtomFloat,   // float atomics that return a value: opt-in (VGPU_HOST_GPU_ATOMICS=1)
  AtomSwap,    // CAS, CAST.SPIN, EXCH: opt-in; the card does not agree with itself from run to run on programs built on these (09 doc, M2)
  Never,       // textures, surfaces, anything not named
};
inline constexpr unsigned kInsnClasses = 11;
inline const char* insn_class_name(InsnClass c) {
  static const char* const n[kInsnClasses] = {"exact", "mufu", "half", "float-red", "call", "extra", "tensor", "atomic-int", "atomic-float", "atomic-swap", "never"};
  return n[static_cast<unsigned>(c)];
}

// The decoder keeps some modifiers joined as nvdisasm prints them ("F32.FTZ.RN", "F16x2.RN"): these look at the dot-separated parts.
template <class Pred>
inline bool any_mod_part(const std::vector<std::string>& mods, Pred pred) {
  for (const std::string& m : mods)
    for (size_t at = 0; at <= m.size();) {
      const size_t dot = std::min(m.find('.', at), m.size());
      if (dot > at && pred(m.substr(at, dot - at))) return true;
      at = dot + 1;
    }
  return false;
}
inline bool mods_have(const std::vector<std::string>& mods, const char* part) {
  return any_mod_part(mods, [&](const std::string& p) { return p == part; });
}
// A float operand type: F16x2, F32, F32x4, F64, BF16x2 ...
inline bool mods_float_typed(const std::vector<std::string>& mods) {
  return any_mod_part(mods, [](const std::string& p) { return p.rfind("F16", 0) == 0 || p.rfind("F32", 0) == 0 || p.rfind("F64", 0) == 0 || p.rfind("BF16", 0) == 0; });
}
// A field the decoder could not name prints as "(9)": an instruction that carries one is not known.
inline bool mods_undecoded(const std::vector<std::string>& mods) {
  return any_mod_part(mods, [](const std::string& p) { return p[0] == '('; });
}

inline InsnClass relaxed_class(const std::string& mnemonic, const std::vector<std::string>& mods) {
  using C = InsnClass;
  const auto is_one_of = [&](std::initializer_list<const char*> names) {
    for (const char* n : names)
      if (mnemonic == n) return true;
    return false;
  };
  if (mnemonic.empty() || mods_undecoded(mods)) return C::Never;
  // Reductions and atomics first, by what they do: the exact set's own check of RED does not see the joined float types.
  if (is_one_of({"RED", "REDG"})) {
    if (mods_have(mods, "CAS") || mods_have(mods, "CAST") || mods_have(mods, "SPIN") || mods_have(mods, "EXCH")) return C::AtomSwap;
    if (mods_float_typed(mods)) return C::FloatRed;
    if (mods_have(mods, "INC") || mods_have(mods, "DEC") || mods_have(mods, "POPC")) return C::AtomInt;
    return mnemonic == "RED" ? C::Exact : C::Extra;   // (REDG is sm_90's spelling: the exact set never named it)
  }
  if (is_one_of({"ATOM", "ATOMG", "ATOMS"})) {
    if (mods_have(mods, "ARRIVE")) return C::Never;   // sm_80's mbarrier.arrive
    if (mods_have(mods, "CAS") || mods_have(mods, "CAST") || mods_have(mods, "SPIN") || mods_have(mods, "EXCH")) return C::AtomSwap;
    return mods_float_typed(mods) ? C::AtomFloat : C::AtomInt;
  }
  if (exact_opcode_allowed(mnemonic, mods)) return C::Exact;
  if (mnemonic == "MUFU") return C::Mufu;
  if (is_one_of({"HADD2", "HMUL2", "HFMA2", "HSETP2", "HSET2", "HMNMX2"})) return C::Half;
  if (is_one_of({"CALL", "RET"})) return C::Call;
  if (is_one_of({"HMMA", "IMMA", "DMMA"})) return C::Tensor;
  if (is_one_of({"IDP", "VABSDIFF", "VABSDIFF4", "VIADDMNMX", "VIMNMX3", "FMNMX3", "F2IP", "FSWZADD", "REDUX", "LDGSTS", "LDGDEPBAR", "LDSM", "MOVM", "FFMA2",
                 "FMUL2", "FADD2"}))
    return C::Extra;
  return C::Never;
}

// A call stays inside the kernel when it is relative, to a label, and the label is an address of the kernel's own code section [lo, hi]. A call through a
// register or to an absolute address (another section, a function pointer) leaves the code this scan has seen.
inline bool call_stays_inside(bool relative, bool to_label, int64_t target, uint64_t lo, uint64_t hi) {
  return relative && to_label && target >= 0 && static_cast<uint64_t>(target) >= lo && static_cast<uint64_t>(target) <= hi;
}

// The classes a kernel holds, as bits, and what the mode's switches make of them.
using ClassMask = uint32_t;
inline constexpr ClassMask class_bit(InsnClass c) { return ClassMask{1} << static_cast<unsigned>(c); }

struct RelaxedSwitches {
  bool atomics = false;     // VGPU_HOST_GPU_ATOMICS=1: float atomics with a returned value, CAS, CAST.SPIN and EXCH
  ClassMask off = 0;        // VGPU_HOST_GPU_RELAXED_OFF: classes switched off for an experiment
};

// VGPU_HOST_GPU_RELAXED_OFF=mufu,tensor,half,call,atomics,red,extra (a comma list). False, with the bad name in *bad, for anything else.
inline bool parse_relaxed_off(const char* text, ClassMask* off, std::string* bad = nullptr) {
  *off = 0;
  if (!text) return true;
  const std::string t(text);
  for (size_t at = 0; at <= t.size();) {
    const size_t comma = std::min(t.find(',', at), t.size());
    const std::string name = t.substr(at, comma - at);
    at = comma + 1;
    if (name.empty()) continue;
    if (name == "mufu") *off |= class_bit(InsnClass::Mufu);
    else if (name == "tensor") *off |= class_bit(InsnClass::Tensor);
    else if (name == "half") *off |= class_bit(InsnClass::Half);
    else if (name == "call") *off |= class_bit(InsnClass::Call);
    else if (name == "atomics") *off |= class_bit(InsnClass::AtomInt) | class_bit(InsnClass::AtomFloat) | class_bit(InsnClass::AtomSwap);
    else if (name == "red") *off |= class_bit(InsnClass::FloatRed);
    else if (name == "extra") *off |= class_bit(InsnClass::Extra);
    else {
      if (bad) *bad = name;
      return false;
    }
  }
  return true;
}

// Why relaxed mode keeps a kernel in the simulator, by the classes of its code. The first that applies, in this order.
enum class RelaxedRefusal : uint8_t {
  None,      // admitted
  Never,     // an instruction outside every class
  Off,       // a class switched off with VGPU_HOST_GPU_RELAXED_OFF
  Atomics,   // a float atomic with a returned value, a CAS or an EXCH, without VGPU_HOST_GPU_ATOMICS=1
};
inline RelaxedRefusal relaxed_refusal(ClassMask classes, const RelaxedSwitches& sw) {
  if (classes & class_bit(InsnClass::Never)) return RelaxedRefusal::Never;
  if (classes & sw.off) return RelaxedRefusal::Off;
  if (!sw.atomics && (classes & (class_bit(InsnClass::AtomFloat) | class_bit(InsnClass::AtomSwap)))) return RelaxedRefusal::Atomics;
  return RelaxedRefusal::None;
}

// ---- residency (the design for a card copy that outlives a launch) ---------------------------------------------------------

enum class Where : uint8_t {
  Host,   // the simulator's bytes are the truth; a card copy, if any, is stale
  Card,   // the card's bytes are the truth; the simulator's are stale (never read them)
  Both,   // the two are equal
};

// The residency of the chunks of one allocation. A new allocation is Host (zeros live in the simulator's sparse table), a
// simulator write makes a chunk Host, a launch on the card makes every chunk of every allocation it may reach Card (it may
// write any of them: pointers inside memory cannot be followed), a push makes Host chunks Both, and a pull makes Card chunks
// Both. The protocol's whole content: a reader or writer on one side first calls the "need" function for its range and
// copies what it returns; a copy in the other direction is never needed for correctness, only for speed.
class Residency {
 public:
  explicit Residency(uint64_t chunks = 0) : state_(chunks, Where::Host) { n_[0] = chunks; }
  uint64_t chunks() const { return state_.size(); }
  Where at(uint64_t c) const { return state_[c]; }

  // The simulator is about to write chunks [first, last]: anything the card holds of them goes stale. The chunks must be
  // current in the simulator first, unless the whole chunk is overwritten (the caller says so with `whole`).
  // Returns the chunks that have to be pulled from the card before the write (partial writes only).
  std::vector<uint64_t> sim_write(uint64_t first, uint64_t last, bool whole) {
    std::vector<uint64_t> pull;
    for (uint64_t c = first; c <= last && c < state_.size(); ++c) {
      if (state_[c] == Where::Card && !whole) pull.push_back(c);
      put(c, Where::Host);
    }
    return pull;
  }
  // The simulator is about to read chunks [first, last]: the Card ones have to come back first (they become Both).
  std::vector<uint64_t> sim_read(uint64_t first, uint64_t last) {
    std::vector<uint64_t> pull;
    for (uint64_t c = first; c <= last && c < state_.size(); ++c)
      if (state_[c] == Where::Card) {
        pull.push_back(c);
        put(c, Where::Both);
      }
    return pull;
  }
  // A launch on the card is about to run: the Host chunks have to be pushed (they become Both). Every chunk is then Card,
  // for the launch may write any of them; call `card_launched` after it.
  std::vector<uint64_t> card_prepare() {
    std::vector<uint64_t> push;
    for (uint64_t c = 0; c < state_.size(); ++c)
      if (state_[c] == Where::Host) {
        push.push_back(c);
        put(c, Where::Both);
      }
    return push;
  }
  void card_launched() {
    std::fill(state_.begin(), state_.end(), Where::Card);
    n_[0] = n_[2] = 0;
    n_[1] = state_.size();
  }
  // After a launch whose written chunks are known (a host-side diff, or a kernel whose stores are bounded).
  void card_launched_writing(uint64_t first, uint64_t last) {
    for (uint64_t c = first; c <= last && c < state_.size(); ++c) put(c, Where::Card);
  }
  // O(1): the counts are kept by every change of a chunk (the gate asked for them after every access to a card-valid allocation, in a loop over the chunks, which
  // made PolyBench 2DCONV with VGPU_HOST_GPU_VERIFY=1 and 128 MiB live run for more than 8 minutes, Round sp42 finding 6).
  uint64_t count(Where w) const { return n_[static_cast<unsigned>(w)]; }
  // Chunks whose card copy is valid (Card or Both): an allocation with any is kept out of the memory manager's lookup memories.
  uint64_t card_valid() const { return n_[1] + n_[2]; }
  // Whether an access of [first, last] needs the gate to change anything: a read, when a chunk of it is the card's alone (it has to be pulled); a write, when a chunk
  // of it is not the simulator's alone (a card copy goes stale, or has to be pulled first). Reads of Host and Both chunks, and writes to Host chunks, change nothing.
  bool access_changes_state(uint64_t first, uint64_t last, bool write) const {
    for (uint64_t c = first; c <= last && c < state_.size(); ++c)
      if (write ? state_[c] != Where::Host : state_[c] == Where::Card) return true;
    return false;
  }
  // The chunks a launch on the card has to push (the simulator's bytes are the truth for them).
  std::vector<uint64_t> host_chunks() const {
    std::vector<uint64_t> out;
    if (n_[0] == 0) return out;
    for (uint64_t c = 0; c < state_.size(); ++c)
      if (state_[c] == Where::Host) out.push_back(c);
    return out;
  }
  // A launch the simulator runs: it may read or write anything, so the Card chunks come back first and every chunk is then Host.
  std::vector<uint64_t> sim_launch() {
    std::vector<uint64_t> pull;
    for (uint64_t c = 0; c < state_.size(); ++c)
      if (state_[c] == Where::Card) pull.push_back(c);
    std::fill(state_.begin(), state_.end(), Where::Host);
    n_[1] = n_[2] = 0;
    n_[0] = state_.size();
    return pull;
  }
  // The same, for the chunks a simulator launch is known not to write (not used by the runtime: pointers inside memory cannot be followed).
  void set(uint64_t c, Where w) { put(c, w); }

 private:
  void put(uint64_t c, Where w) {
    --n_[static_cast<unsigned>(state_[c])];
    ++n_[static_cast<unsigned>(w)];
    state_[c] = w;
  }
  std::vector<Where> state_;
  uint64_t n_[3] = {0, 0, 0};   // chunks per Where (Host, Card, Both)
};

// Chunk indices to [first, count) runs, for one copy call per run of adjacent chunks.
inline std::vector<std::pair<uint64_t, uint64_t>> chunk_runs(const std::vector<uint64_t>& idx) {
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (uint64_t c : idx) {
    if (!out.empty() && out.back().first + out.back().second == c) ++out.back().second;
    else out.emplace_back(c, 1);
  }
  return out;
}

// The chunks a host access of [addr, addr + len) touches in the allocation [base, base + size), clamped to the allocation, and
// whether the access covers its first and its last chunk completely (a write that does not needs the chunk's old bytes first).
struct ChunkSpan {
  bool any = false;
  uint64_t first = 0, last = 0;
  bool first_whole = false, last_whole = false;
};
inline ChunkSpan chunk_span(uint64_t base, uint64_t size, uint64_t addr, uint64_t len, uint64_t chunk_bytes = 64 * 1024) {
  ChunkSpan s;
  if (!len || !size || addr >= base + size || addr + len <= base) return s;
  const uint64_t lo = addr < base ? 0 : addr - base;
  const uint64_t hi = std::min<uint64_t>(addr + len - base, size);   // exclusive
  s.any = true;
  s.first = lo / chunk_bytes;
  s.last = (hi - 1) / chunk_bytes;
  const auto chunk_end = [&](uint64_t c) { return std::min<uint64_t>((c + 1) * chunk_bytes, size); };
  s.first_whole = lo <= s.first * chunk_bytes && hi >= chunk_end(s.first);
  s.last_whole = lo <= s.last * chunk_bytes && hi >= chunk_end(s.last);
  return s;
}

// ---- the size rule ---------------------------------------------------------------------------------------------------------

// Measured on a g5 rig (A10G, PCIe 4.0): 09 doc, M3. One launch and its synchronisation 7 us; contiguous pageable copies 12 GB/s
// in and 10 GB/s out (1 MiB and up; 64 KiB chunk-at-a-time copies 4.4 and 4.0 GB/s, gathered through pinned staging 5.8).
struct Costs {
  double launch_s = 7e-6;
  double push_bytes_per_s = 12e9;
  double pull_bytes_per_s = 10e9;
  double kernel_s_default = 100e-6;   // before the card has run this kernel once
  double margin = 2.0;                // offload only when the simulator is expected to take this many times the card path
  double sim_warp_instr_per_s = 100e6;   // a lower bound of the simulator's speed in warp instructions per second (tens to ~150 M per thread in the benchmarks)
  double min_sim_s = 200e-6;          // never below this, whatever the arithmetic says (timer noise, the estimate's error)
  // The one-off cost of the first card launch: loading the driver, cuInit, the primary context and the window reservation. Measured on an A10G at 0.23 s of
  // process wall time (docs/performance.md "Round sp42": the native process of a trivial program also takes 0.24 s). 0.25 rounds it up. VGPU_HOST_GPU_START_COST
  // sets it in seconds; 0 charges nothing (the rule before sp43: the driver starts at the first launch the per-launch rule accepts).
  double start_cost_s = 0.25;
};

struct Estimate {
  double sim_s = 0;                   // what the simulator is expected to need
  std::optional<double> card_kernel_s;
  uint64_t push_bytes = 0;
  uint64_t pull_bytes = 0;
  double start_s = 0;                 // the driver start this launch would pay (the first card launch of the process); 0 once the driver is up
};

inline double card_path_seconds(const Estimate& e, const Costs& c) {
  return e.start_s + c.launch_s + e.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(e.push_bytes) / c.push_bytes_per_s +
         static_cast<double>(e.pull_bytes) / c.pull_bytes_per_s;
}

inline bool should_offload(const Estimate& e, const Costs& c) {
  return e.sim_s >= c.min_sim_s && e.sim_s > c.margin * card_path_seconds(e, c);
}

// The rule for a launch, charging the driver start while it has not been paid (sp43). Before it, the rule judged a launch against the card path ALONE and a
// program whose launches each looked worth a card paid the 0.23 s start for a saving of milliseconds (all 8 Rodinia programs: 0.04x to 0.29x), and some paid
// it with every launch kept in the simulator (the start was kicked off at the first big-enough launch and awaited at exit). The driver now starts only when
// this function says a launch goes, and it says so when EITHER
//   * the launch alone repays the start: the simulator is expected to take more than `margin` times the card path INCLUDING the start (a single large launch:
//     PolyBench SYRK, SYR2K, 2MM, FDTD-2D), or
//   * the launch is worth a card with the driver up (should_offload without the start) AND the program has already spent at least `start_cost_s` of simulator
//     time in launches that could have gone to the card (`eligible_sim_spent_s`: the sum over every launch of an exact-set kernel that stayed in the simulator).
//     This is the ski-rental rule: the start is bought only after as much has been spent renting as buying costs, so a start that turns out useless costs at most
//     the time the program had already spent, never more than doubling it, and a program that never spends that much never starts the driver.
// With the driver up the start is paid and only should_offload applies. start_cost_s <= 0 is the old rule.
inline bool should_offload_with_start(const Estimate& e, const Costs& c, bool driver_started, double eligible_sim_spent_s) {
  if (!should_offload(e, c)) return false;                      // not worth a card even with the driver up
  if (driver_started || c.start_cost_s <= 0) return true;
  Estimate cold = e;
  cold.start_s = c.start_cost_s;
  return should_offload(cold, c) || eligible_sim_spent_s >= c.start_cost_s;
}

// ---- start-up of the real driver ---------------------------------------------------------------------------------------------------
// Loading the driver and making its context costs 0.15 to 0.35 s (09 doc, section 5; 0.23 s on the A10G of docs/performance.md "Round sp42"). Nothing touches
// the driver, and no thread is started for it, until should_offload_with_start has decided that a launch goes to the card: the start is then made at that launch
// and the launch waits for it (Costs::start_cost_s is part of that decision). A program that never has such a launch never loads the driver and does not wait for it
// at exit. (Until sp42 the start was kicked off, on a thread, at the first launch of an exact-set kernel on a grid of 32 warps or more and awaited at exit, so
// every program with one such launch paid it, offloaded or not.)

// An estimate for a launch the simulator has not run yet: every warp of the grid executes the kernel's static instruction count, at the
// simulator's rate. It is NOT a lower bound: a thread that leaves at a bounds check runs a few instructions (PolyBench 3DCONV, a stencil
// whose grid covers the array and whose threads mostly exit, simulated in 3 ms and estimated at 60 ms), and a loop adds to the count.
// The runtime uses it only when asked (VGPU_HOST_GPU_STATIC=1); by default a kernel's first launch is the simulator's.
inline double static_sim_seconds(uint64_t blocks, uint32_t threads_per_block, uint64_t static_instrs, const Costs& c) {
  const uint64_t warps_per_block = (threads_per_block + 31) / 32;
  return static_cast<double>(blocks * warps_per_block * static_instrs) / c.sim_warp_instr_per_s;
}

// What the simulator took for a kernel before, by kernel, per block of the grid. The first two launches of a kernel run in the simulator: the first one
// pays decoding and set-up that the later ones do not (PolyBench 3DCONV's first launch took far longer than the 252 after it, and an estimate from it sent
// 253 launches of 3 ms to a card path of 100 ms: 38x slower), so the estimate is the smallest per-block time seen, which errs towards the simulator, as the
// rule should. It is available after two samples.
//
// One exception, for the launches that cost the most (sp15): a first sample that took kOneSampleMinSeconds or more is a launch whose simulator time is far above any
// decode or set-up cost, and running the kernel a second time in the simulator just to learn what it already showed would cost as much again. Such a kernel
// has an estimate after ONE sample, for at most kOneSampleOffloads offloaded launches (the estimate may still hold set-up time, which would blind the
// offload_paid_off check below): after that it needs a second, simulated sample like any other kernel, and the ban rules judge the card path against that.
inline constexpr double kOneSampleMinSeconds = 0.25;
inline constexpr unsigned kOneSampleOffloads = 2;

class SimTimeTable {
 public:
  void record(const std::string& kernel, uint64_t blocks, double seconds) {
    Entry& e = table_[kernel];
    const double per_block = blocks ? seconds / static_cast<double>(blocks) : seconds;
    if (!e.samples) e.first_total_s = seconds;
    e.per_block_s = e.samples ? std::min(e.per_block_s, per_block) : per_block;
    ++e.samples;
  }
  std::optional<double> predict(const std::string& kernel, uint64_t blocks) const {
    const auto it = table_.find(kernel);
    if (it == table_.end()) return std::nullopt;
    const Entry& e = it->second;
    const bool one_sample_ok = e.samples == 1 && e.first_total_s >= kOneSampleMinSeconds && e.one_sample_offloads < kOneSampleOffloads;
    if (e.samples < kSamplesNeeded && !one_sample_ok) return std::nullopt;
    return e.per_block_s * static_cast<double>(std::max<uint64_t>(blocks, 1));
  }
  // An offloaded launch used an estimate from `kernel`'s samples: counts against the one-sample allowance.
  void note_offload(const std::string& kernel) {
    const auto it = table_.find(kernel);
    if (it != table_.end() && it->second.samples < kSamplesNeeded) ++it->second.one_sample_offloads;
  }
  unsigned samples(const std::string& kernel) const {
    const auto it = table_.find(kernel);
    return it == table_.end() ? 0 : it->second.samples;
  }
  static constexpr unsigned kSamplesNeeded = 2;

 private:
  struct Entry {
    double per_block_s = 0;
    double first_total_s = 0;
    unsigned samples = 0;
    unsigned one_sample_offloads = 0;
  };
  std::map<std::string, Entry> table_;
};

// ---- the first launch of a kernel (docs/perf-design/09-host-gpu-spike.md 6, step 2) --------------------------------------------------
// The history rule needs two simulator launches of a kernel, so a program that launches each kernel once (PolyBench GEMM, SYRK, SYR2K, 3MM) was never
// offloaded. A static count of the code cannot say how long threads run (3DCONV, a stencil with no loop whose threads mostly leave at a bounds check, was
// 38x slower with VGPU_HOST_GPU_STATIC=1), so the first launch of a kernel goes to the card only when ALL of these hold:
//  * the kernel has a loop (a branch back to an earlier instruction): threads cannot all leave at a bounds check, they have work that grows with the code's loop;
//  * the grid covers at least kFirstLaunchBlocksPerSm blocks per multiprocessor: a grid that does not fill the machine is small however long a thread runs;
//  * the live bytes (the most the launch can move each way) are at most kFirstLaunchMaxBytes: a mistaken first launch is bounded;
//  * the static estimate beats the card path by the usual margin, with the pushes and pulls of the whole live memory.
// Only the kernel's first launch is decided this way: the next ones are the simulator's until its history exists, as before.
inline constexpr uint64_t kFirstLaunchBlocksPerSm = 8;
inline constexpr uint64_t kFirstLaunchMaxBytes = 256ull << 20;

// Whether a branch goes back to its own instruction or an earlier one. `branches` holds (pc, target) of every branch of the kernel's code.
inline bool has_back_edge(const std::vector<std::pair<uint64_t, uint64_t>>& branches) {
  for (const auto& [pc, target] : branches)
    if (target <= pc) return true;
  return false;
}

struct FirstLaunch {
  uint64_t blocks = 0;
  uint32_t threads_per_block = 0;
  uint64_t static_instrs = 0;
  bool back_edge = false;
  uint32_t multiprocessors = 0;
  uint64_t live_bytes = 0;
};

inline bool first_launch_offload(const FirstLaunch& f, const Costs& c, bool driver_started = true, double eligible_sim_spent_s = 0) {
  if (!f.back_edge || f.multiprocessors == 0 || f.blocks < kFirstLaunchBlocksPerSm * f.multiprocessors || f.live_bytes > kFirstLaunchMaxBytes) return false;
  Estimate e;
  e.sim_s = static_sim_seconds(f.blocks, f.threads_per_block, f.static_instrs, c);
  e.push_bytes = f.live_bytes;
  e.pull_bytes = f.live_bytes;
  return should_offload_with_start(e, c, driver_started, eligible_sim_spent_s);
}

// A safety net for any estimate that turns out wrong: an offloaded launch that took longer on the card path (copies, launch, copy back)
// than the simulator was expected to need ends the offloading of that kernel. One bad launch is the most it can cost.
inline bool offload_paid_off(double card_total_s, double sim_estimate_s) { return card_total_s <= sim_estimate_s; }

// A second safety net, by traffic (docs/perf-design/09-host-gpu-spike.md 7.3; docs/perf-design/rounds/08-rounds-sp11-sp15.md "Round sp14"). With a persistent card copy, a kernel that
// alternates with simulator launches on the same buffers gives the copy up and gets it pushed again every time. A kernel is banned when kTrafficLaunches
// offloaded launches IN A ROW each pushed more than kTrafficSharePercent percent of the live bytes AND more than kTrafficMinBytes, and each one's card path
// took more than kTrafficSlowShare of the simulator's estimate. All three conditions are needed: replayed over the 64 saved traces (residency_model.py), the share
// and the size alone also catch GRAMSCHM, whose offloaded launches are worth 3x their traffic (banning it loses 29%), and the share keeps base64e (8.4% of live bytes
// per push, 8.06 MiB) out. The kernel stays in the simulator for the rest of the process; a ban only moves a launch back, so results do not change.
// Measured on the traces the ban fires on LU only, and late: it is a net, not the fix for LU's 18% against write-through.
inline constexpr unsigned kTrafficLaunches = 2;
inline constexpr unsigned kTrafficSharePercent = 25;
inline constexpr uint64_t kTrafficMinBytes = 12ull << 20;
inline constexpr double kTrafficSlowShare = 0.5;

class TrafficTable {
 public:
  // One offloaded launch of `kernel`: it pushed `push_bytes` of `live_bytes`, its card path took `card_total_s`, the simulator was expected to need
  // `sim_estimate_s`. `timing_valid` is false for a kernel's first card launch, whose path includes the module load: then only the traffic counts.
  // Returns true when the kernel must go back to the simulator for good.
  bool record(const std::string& kernel, uint64_t push_bytes, uint64_t live_bytes, double card_total_s, double sim_estimate_s, bool timing_valid = true) {
    unsigned& heavy = heavy_[kernel];
    const bool big = push_bytes > kTrafficMinBytes && live_bytes > 0 && push_bytes * 100 > live_bytes * kTrafficSharePercent;
    const bool slow = !timing_valid || card_total_s > kTrafficSlowShare * sim_estimate_s;
    heavy = (big && slow) ? heavy + 1 : 0;
    return heavy >= kTrafficLaunches;
  }

 private:
  std::map<std::string, unsigned> heavy_;   // per kernel: offloaded launches in a row that were heavy
};

}  // namespace vgpu::hostgpu
