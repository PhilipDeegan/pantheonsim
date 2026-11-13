// Exact parallel launches: the shadow of device memory a trial of one launch is run under, and its undo log
// (vgpu/memory.hpp, "exact parallel launches"; docs/exact-threads.md has the argument for why a trial that commits
// left the bits of the serial run).
//
// The shadow has two levels. A page state stands for 4 KiB of the device's address range [va_base_, high_water_va_):
//     0                 no block has touched the page
//     b + 1             block b has read in it; no other block has touched it; nothing is noted by word
//     kMany             more than one block has read in it, and none has written in it; nothing is noted by word
//     kWords            the page is noted by word (below), and every word says what was done to its bytes
//     kWordsCoarse      the same, after one block read in the page by page state and then wrote in it: its reads were
//                       not noted by word, so every word started as "read by that block" (more than it read)
// A page moves 0 -> one reader -> many readers by compare-and-swap, or 0 -> kWords (the first block that writes in it,
// or, where the trial's plan says by word, the first that touches it), or (its only reader, about to write) its own
// reader mark -> kWordsCoarse; kMany, kWords and kWordsCoarse are final. A write is only made in a page noted by word.
// So a page some block read by page state is written by no other block: that block would have to bring the page to a
// word state, which only the page's sole reader can do, and it fills the words with its own reader mark first.
//
// In a page noted by word, one 32-bit word stands for four bytes:
//     0                 no block has touched the bytes
//     b + 1             block b has read them, and no other block has touched them
//     kMany             more than one block has read them, and none has written them
//     kWrite | (b + 1)  block b has written them (and may have read them); no other block has touched them
// A word only moves up that list (0 -> one reader -> many readers; 0 or its own reader mark -> its writer), by
// compare-and-swap, and a block makes its access only after the page and the word allow it. So for any two blocks and
// any byte:
//   * both read it: nothing to tell apart;
//   * one writes it and the other reads or writes it: whichever came second to the page or to the word found a state
//     it may not enter, and stopped the trial BEFORE touching the byte.
// A stopped trial is undone from the bytes each block kept when it first wrote a word (it owned the word then, so
// what it kept is what the launch started with), and the launch is run again in order.
//
// The words of a page noted by word are kept sixteen at a time while one block alone has touched them (round sp58): a
// line is 64 bytes of device memory, and its record is one 64-bit value
//     0                                  no block has touched the line's sixteen words
//     b + 1 | read bits | written bits   block b alone has touched them: a bit per word it read, a bit per word it wrote
//     kLineByWord                        the sixteen words (above) say what was done; final
//     kLineBusy                          a block is writing the sixteen words out from the record (a few stores long), or the
//                                        line's block is keeping its 64 bytes before its first write in it; others wait
// The record of one block is the sixteen word marks 0, b + 1 and kWrite | (b + 1) written down in 64 bits, and a block
// moves it by one compare-and-swap, which is the single-word moves of several of its words made at the same instant.
// A second block that comes to the line reads from the record exactly which words the first read and wrote: where its
// own access may not be made by the word rules it stops the trial; otherwise it writes the marks out to the words
// (kLineBusy while it does, so that no block sees half of them) and the line is noted by word from then on. Nothing is
// noted that the words would not note, so a line never makes a stop of its own.
//
// What a block keeps for undoing in a line it alone holds is the whole line, once, at its first write in it: the record
// is kLineBusy while the 64 bytes are copied, so no other block can come to the line and write in it meanwhile, and no
// block had touched it before; the copy is what the trial started with in all sixteen words. The block's later writes in
// the line keep nothing. (Another block that comes to the line afterwards keeps its own words by the word rules; those
// bytes are what the trial started with too, so the order of putting back stays free.)
//
// A stop at a page state, or at a word of a kWordsCoarse page that says "read", may be the note's fault and not the
// kernel's (the page says more was read than was). Such a stop is marked coarse and names the allocation; the caller
// may run the trial again with that allocation noted by word from the start (SpecPlan), which is the exact note.
#include "vgpu/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <sys/mman.h>

namespace vgpu {

// The windows of a line's record, for the race test only (tests/unit/test_exact_lines_race.cpp, which compiles this file with
// VGPU_SPEC_TEST_HOOKS): `spec_test::hook` is called at each (a block about to write another's record out, and having taken it;
// a block half way through writing it out; a block about to keep its line's bytes, having taken it; a block about to note more
// in its own record, or to take an untouched line as its first reader; a block waiting at a busy line), so that the test can hold one thread inside a window while another comes to the line. Without the definition
// nothing is compiled in.
#ifdef VGPU_SPEC_TEST_HOOKS
namespace spec_test {
enum : int { kOutBefore = 1, kOutBusy = 2, kKeepBusy = 3, kWaitBusy = 4, kOwnBefore = 5, kOutHalf = 6, kFirstBefore = 7 };
void (*hook)(int at) = nullptr;
}  // namespace spec_test
#define VGPU_SPEC_HOOK(at) (spec_test::hook ? spec_test::hook(spec_test::at) : void())
#else
#define VGPU_SPEC_HOOK(at) ((void)0)
#endif

class MemoryManager::Spec {
 public:
  // A mark of the shadow: 32 bits, read and moved only by the atomic builtins.
  struct Word {
    uint32_t v;
    uint32_t load(std::memory_order) const { return __atomic_load_n(&v, __ATOMIC_ACQUIRE); }
    void store(uint32_t x, std::memory_order) { __atomic_store_n(&v, x, __ATOMIC_RELEASE); }
    bool compare_exchange_weak(uint32_t& expected, uint32_t desired, std::memory_order) {
      return __atomic_compare_exchange_n(&v, &expected, desired, true, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
    bool compare_exchange_strong(uint32_t& expected, uint32_t desired, std::memory_order) {
      return __atomic_compare_exchange_n(&v, &expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
  };
  // The record of a line (sixteen words of a page noted by word; the comment at the top of the file): the block's mark in the low
  // half, then a bit per word it read, then a bit per word it wrote.
  struct Line {
    uint64_t v;
    uint64_t load() const { return __atomic_load_n(&v, __ATOMIC_ACQUIRE); }
    void store(uint64_t x) { __atomic_store_n(&v, x, __ATOMIC_RELEASE); }
    bool take(uint64_t& expected, uint64_t desired) {
      return __atomic_compare_exchange_n(&v, &expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
  };
  static constexpr uint64_t kLineWords = 16, kPageLines = 64;
  static constexpr uint32_t kLineByWord = 0xFFFFFFFEu, kLineBusy = 0xFFFFFFFDu;   // (no block's mark, as the page states)
  struct Worker;
  // A line is kLineBusy for sixteen stores, or for a copy of 64 bytes (which may end in a stop: the waiting block then stops too).
  // (After a few looks the thread gives its CPU up: the block that holds the line may have lost its own.)
  void wait_busy(Worker& w) {
    if (stopped.load(std::memory_order_relaxed)) throw SpecStop{};
    VGPU_SPEC_HOOK(kWaitBusy);
    if (++w.waits % 64 == 0) {
      std::this_thread::yield();
      return;
    }
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
  }
  static uint32_t owner(uint64_t line) { return static_cast<uint32_t>(line); }
  static uint32_t read_bits(uint64_t line) { return static_cast<uint32_t>(line >> 32) & 0xFFFFu; }
  static uint32_t written_bits(uint64_t line) { return static_cast<uint32_t>(line >> 48); }
  static uint64_t as_read(uint32_t bits) { return uint64_t{bits} << 32; }
  static uint64_t as_written(uint32_t bits) { return uint64_t{bits} << 48; }
  // The bits of words [first, last] of a line (0 <= first <= last <= 15).
  static uint32_t bits_of(uint64_t first, uint64_t last) { return (0xFFFFu >> (15 - (last - first))) << first; }
  static constexpr uint32_t kMany = 0xFFFFFFFFu, kWrite = 0x80000000u;
  // Page states only (no block's mark: run_exact tries fewer than 0x7FFFFFF0 blocks).
  static constexpr uint32_t kWords = 0xFFFFFFFEu, kWordsCoarse = 0xFFFFFFFDu;
  static bool by_words(uint32_t state) { return state == kWords || state == kWordsCoarse; }
  static constexpr uint64_t kGranule = 4;          // bytes a word stands for
  static constexpr uint64_t kPageWords = 1024;     // words of a shadow page: 4 KiB of device memory, 4 KiB of shadow
  static constexpr uint64_t kPageBytes = kGranule * kPageWords;
  // What stands for a page noted by word: its lines' records, and the words of the lines that more than one block came to (the rest of
  // `words` is never touched, so a page whose blocks keep to their own lines costs 512 bytes of host memory that is ever written).
  struct Page {
    Line lines[kPageLines];
    Word words[kPageWords];
  };
  static constexpr size_t kPoolPages = 65536;      // shadow pages kept between trials (288 MiB of host memory at most)
  static constexpr uint64_t kKeepLimit = uint64_t{1} << 30;   // bytes one worker may keep for undoing before the trial stops
  static constexpr size_t kSeen = 256;             // pages a worker remembers it may read without a look at the shadow
  static constexpr size_t kRanges = 4;             // runs of words a worker remembers its block has noted
  static constexpr uint64_t kRunWorth = 16;        // ... of at least so many words: a warp's lanes one after another, not one lane's
                                                   // few words (32 lanes' runs would push one another out, and asking costs each of them)
  static constexpr size_t kWordPages = 64;         // pages of words a worker remembers where to find
  static constexpr size_t kCoarseKept = 64;        // addresses of coarse stops kept for the caller

  // A page the worker's block may read with no further note: the page's state was the block's own reader mark (`tag` is the block's)
  // or kMany (`tag` is kMany: any block's, for the rest of the trial) when it was looked at.
  struct Seen {
    uint64_t slot = ~uint64_t{0};
    uint32_t tag = 0;
  };

  // Where the worker keeps its memory of a page: a warp's lanes are often a fixed number of pages apart (rows of a matrix), which
  // the low bits of the page number would send to a few places.
  static size_t seen_at(uint64_t slot) { return static_cast<size_t>((slot * 0x9E3779B97F4A7C15ull) >> 56); }

  // A run of words [g0, g1] (granules of the range) every one of which block `tag` has noted: as its writer (the worker's `wrote`:
  // each word holds the block's writer mark, which nothing changes), or for reading (`read`: each held the block's reader or
  // writer mark or the several-readers mark when it was noted, and from those a word only moves to another of them). The block's
  // next access inside the run has nothing to note.
  struct Run {
    uint64_t g0 = 1, g1 = 0;
    uint32_t tag = 0;   // (no block's: marks start at 1)
  };
  static bool known(const Run* runs, uint32_t tag, uint64_t g0, uint64_t g1) {
    for (size_t i = 0; i < kRanges; ++i)
      if (runs[i].tag == tag && g0 >= runs[i].g0 && g1 <= runs[i].g1) return true;
    return false;
  }
  // Two runs of one block that touch or overlap are one run (every word of either is noted).
  static void learn(Run* runs, uint8_t& next, uint32_t tag, uint64_t g0, uint64_t g1) {
    for (size_t i = 0; i < kRanges; ++i)
      if (runs[i].tag == tag && g0 <= runs[i].g1 + 1 && g1 + 1 >= runs[i].g0) {
        runs[i].g0 = std::min(runs[i].g0, g0);
        runs[i].g1 = std::max(runs[i].g1, g1);
        return;
      }
    runs[next] = {g0, g1, tag};
    next = static_cast<uint8_t>((next + 1) % kRanges);
  }

  struct Worker {
    uint32_t tag = 0;                    // block + 1 of the block it runs now
    std::vector<Page*> fresh;            // zeroed pages ready to install
    std::vector<uint64_t> installed;     // the top-table slots whose pages this worker installed
    std::vector<uint64_t> noted;         // the slots whose page state this worker moved from 0
    std::vector<std::pair<uint64_t, uint64_t>> kept_at;   // (address, length) of each kept piece, in the order kept
    std::vector<uint8_t> kept;                            // their bytes, one after another
    // The last page noted by word that was asked for (a block's accesses are mostly near one another).
    uint64_t last_slot = ~uint64_t{0};
    Page* last_page = nullptr;
    // The allocation the last kept piece was in.
    uint64_t keep_base = 1, keep_end = 0;
    const void* keep_alloc = nullptr;
    uint32_t waits = 0;                  // looks at a busy line (wait_busy)
    SpecStats st;
    Run wrote[kRanges], read[kRanges];
    uint8_t wrote_next = 0, read_next = 0;
    // Pages noted by word the worker has read in, and their words (a page's words stay where they are for the trial).
    struct WordPage {
      uint64_t slot = ~uint64_t{0};
      Page* page = nullptr;
    } pages[kWordPages];
    Seen seen[kSeen];

    // Ready for a trial. The undo log keeps its memory from the last trial (a launch that writes much writes as much the next time).
    void reset() {
      installed.clear();
      noted.clear();
      kept_at.clear();
      kept.clear();
      keep_base = 1, keep_end = 0;
      keep_alloc = nullptr;
      st = SpecStats{};
      forget();
    }
    // What the worker remembers of the shadow (every memory is of marks that are there; when marks are taken away, so is this).
    void forget() {
      tag = 0;
      last_slot = ~uint64_t{0};
      last_page = nullptr;
      for (size_t i = 0; i < kRanges; ++i) wrote[i] = read[i] = Run{};
      wrote_next = read_next = 0;
      for (Seen& e : seen) e = Seen{};
      for (WordPage& e : pages) e = WordPage{};
    }
  };

  ~Spec() {
    for (Page* p : pool) std::free(p);
    if (top) ::munmap(top, top_slots * sizeof(std::atomic<Page*>));
    if (pstate) ::munmap(pstate, top_slots * sizeof(Word));
  }

  // The top tables: a slot per shadow page of the range (its state, and its page of words once it is noted by word), zero pages of
  // the kernel's until a slot is written.
  void size_for(uint64_t base_va, uint64_t bytes) {
    base = base_va;
    span = bytes;
    const uint64_t slots = (bytes / kGranule + kPageWords - 1) / kPageWords + 1;
    if (slots <= top_slots) return;
    if (top) ::munmap(top, top_slots * sizeof(std::atomic<Page*>));
    if (pstate) ::munmap(pstate, top_slots * sizeof(Word));
    top = nullptr;
    pstate = nullptr;
    const uint64_t want = std::max<uint64_t>(slots * 2, 4096);
    top_slots = 0;
    void* m = ::mmap(nullptr, want * sizeof(std::atomic<Page*>), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) throw std::bad_alloc();
    void* m2 = ::mmap(nullptr, want * sizeof(Word), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m2 == MAP_FAILED) {
      ::munmap(m, want * sizeof(std::atomic<Page*>));
      throw std::bad_alloc();
    }
    top = static_cast<std::atomic<Page*>*>(m);   // (zero bytes: null pointers)
    pstate = static_cast<Word*>(m2);             // (zero bytes: untouched pages)
    top_slots = want;
  }

  Page* fresh_page(Worker& w) {
    if (w.fresh.empty()) {
      std::lock_guard<std::mutex> g(pool_mu);
      for (int i = 0; i < 16 && !pool.empty(); ++i) {
        w.fresh.push_back(pool.back());
        pool.pop_back();
      }
    }
    if (!w.fresh.empty()) {
      Page* p = w.fresh.back();
      w.fresh.pop_back();
      return p;
    }
    void* m = std::calloc(1, sizeof(Page));
    if (!m) throw std::bad_alloc();
    return static_cast<Page*>(m);
  }

  // The lines and words of a page whose state is kWords or kWordsCoarse (installed zeroed by whoever comes first), remembered as the worker's last.
  Page* word_page(Worker& w, uint64_t slot) {
    Page* page = top[slot].load(std::memory_order_acquire);
    if (!page) {
      Page* mine = fresh_page(w);
      if (top[slot].compare_exchange_strong(page, mine, std::memory_order_acq_rel)) {
        page = mine;
        w.installed.push_back(slot);
      } else {
        w.fresh.push_back(mine);   // another worker's page is there
      }
    }
    w.last_slot = slot;
    w.last_page = page;
    return page;
  }

  // Whether the page is noted by word from its first touch (the plan of the trial: SpecPlan).
  bool by_word(uint64_t slot) const {
    if (!plan_pages) return true;
    const uint64_t lo = base + slot * kPageBytes, hi = lo + kPageBytes;
    for (const auto& [from, to] : plan_words)
      if (lo < to && from < hi) return true;
    return false;
  }

  [[noreturn, gnu::cold, gnu::noinline]] void refuse(const char* reason) {
    const char* none = nullptr;
    why.compare_exchange_strong(none, reason, std::memory_order_relaxed);
    stopped.store(true, std::memory_order_release);
    throw SpecStop{};
  }
  // A stop the coarseness of a page's note may have made (`addr` is in the page).
  [[noreturn, gnu::cold, gnu::noinline]] void refuse_coarse(uint64_t addr, const char* reason) {
    {
      std::lock_guard<std::mutex> g(pool_mu);
      if (coarse_at.size() < kCoarseKept) coarse_at.push_back(addr);
    }
    coarse.store(true, std::memory_order_release);
    refuse(reason);
  }

  // A block reads in page `slot` and the worker has no memory of the page: the note by the page's state when that is enough (null:
  // nothing more to note), or the page's lines and words when it is noted by word.
  [[gnu::noinline]] Page* read_page(Worker& w, uint64_t slot) {
    const uint32_t tag = w.tag;
    if (slot == w.last_slot) return w.last_page;
    Word& ps = pstate[slot];
    uint32_t v = ps.load(std::memory_order_acquire);
    for (;;) {
      if (v == kMany || v == tag) {
        w.seen[seen_at(slot)] = {slot, v};
        return nullptr;
      }
      if (by_words(v)) break;
      if (v == 0) {
        const uint32_t to = by_word(slot) ? kWords : tag;
        if (!ps.compare_exchange_weak(v, to, std::memory_order_acq_rel)) continue;
        w.noted.push_back(slot);
        ++(to == tag ? w.st.pages_read : w.st.pages_by_word);
        v = to;
        continue;
      }
      // Another block's reader mark: several readers now.
      if (ps.compare_exchange_weak(v, kMany, std::memory_order_acq_rel)) {
        ++w.st.pages_shared;
        v = kMany;
      }
    }
    return word_page(w, slot);
  }

  // A second block has come to line `l` of `page`, whose record `v` is another block's: its marks are written out to the line's
  // words, and the line is noted by word from then on. Nobody looks at the words of a line before its record says kLineByWord,
  // and nobody but the block that put kLineBusy there writes them before that. On return `v` is the record as it is now.
  [[gnu::noinline]] void to_words(Worker& w, Page* page, uint64_t l, uint64_t& v) {
    Line& line = page->lines[l];
    VGPU_SPEC_HOOK(kOutBefore);
    if (!line.take(v, kLineBusy)) return;   // (the record moved: its owner noted more, or another block is writing it out)
    VGPU_SPEC_HOOK(kOutBusy);
    const uint32_t b = owner(v), r = read_bits(v), wr = written_bits(v);
    Word* words = page->words + l * kLineWords;
    for (uint64_t i = 0; i < kLineWords; ++i) {
      words[i].store((wr >> i & 1) ? (b | kWrite) : (r >> i & 1) ? b : 0, std::memory_order_relaxed);
      if (i == kLineWords / 2 - 1) VGPU_SPEC_HOOK(kOutHalf);
    }
    line.store(kLineByWord);
    ++w.st.lines_by_word;
    v = kLineByWord;
  }

  // The words `m` (a bit each) of a line noted by word, for a read: the four states of a word.
  [[gnu::always_inline]] void read_words(Worker& w, Word* words, uint32_t m) {
    const uint32_t tag = w.tag;
    for (; m; m &= m - 1) {
      Word& word = words[__builtin_ctz(m)];
      uint32_t v = word.load(std::memory_order_acquire);
      if (v == kMany || v == tag || v == (tag | kWrite)) [[likely]] continue;
      for (;;) {
        if (v == 0) {
          if (word.compare_exchange_weak(v, tag, std::memory_order_acq_rel)) {
            ++w.st.words_read_first;
            break;
          }
        } else if (v == kMany || v == tag || v == (tag | kWrite)) {
          break;
        } else if (v & kWrite) {
          refuse("a block read what another block wrote");   // (a writer's mark is always exact: no coarse stop here)
        } else if (word.compare_exchange_weak(v, kMany, std::memory_order_acq_rel)) {
          ++w.st.words_read_shared;
          break;
        }
      }
    }
  }

  // A read of words `m` of line `l` found the line another block's, or being written out (`v` is its record).
  [[gnu::noinline]] void read_line_met(Worker& w, Page* page, uint64_t l, uint32_t m, uint64_t v) {
    Line& line = page->lines[l];
    for (;;) {
      const uint32_t o = owner(v);
      if (o == kLineByWord) return read_words(w, page->words + l * kLineWords, m);
      if (o == kLineBusy) {
        wait_busy(w);
        v = line.load();
        continue;
      }
      if (v == 0 || o == w.tag) return read_line(w, page, l, m);   // (cannot be: the caller saw another block's; said, not assumed)
      // Another block's line. A word of the read that it wrote stops the trial as the word would (its written bit was set before
      // its store, and is never cleared); otherwise the two blocks' marks go to the words.
      if (m & written_bits(v)) refuse("a block read what another block wrote");
      to_words(w, page, l, v);
    }
  }

  // A block reads words `m` of line `l` of a page noted by word.
  [[gnu::always_inline]] void read_line(Worker& w, Page* page, uint64_t l, uint32_t m) {
    Line& line = page->lines[l];
    uint64_t v = line.load();
    const uint32_t tag = w.tag;
    for (;;) {
      if (owner(v) == tag) [[likely]] {
        const uint32_t fresh = m & ~(read_bits(v) | written_bits(v));
        if (!fresh) return;
        VGPU_SPEC_HOOK(kOwnBefore);
        if (!line.take(v, v | as_read(fresh))) continue;   // (it moves under its owner only when another block writes it out)
        w.st.words_read_first += static_cast<uint64_t>(__builtin_popcount(fresh));
        return;
      }
      if (v == 0) {
        VGPU_SPEC_HOOK(kFirstBefore);
        if (!line.take(v, uint64_t{tag} | as_read(m))) continue;
        w.st.words_read_first += static_cast<uint64_t>(__builtin_popcount(m));
        return;
      }
      if (owner(v) == kLineByWord) return read_words(w, page->words + l * kLineWords, m);
      return read_line_met(w, page, l, m, v);
    }
  }

  // The words `m` of a line noted by word, for a write: the bits of those the block took just now (their bytes are to be kept).
  [[gnu::always_inline]] uint32_t write_words(Worker& w, Page* page, uint64_t slot, uint64_t l, uint32_t m) {
    const uint32_t tag = w.tag, mine = tag | kWrite;
    Word* words = page->words + l * kLineWords;
    uint32_t taken = 0;
    for (; m; m &= m - 1) {
      const unsigned i = static_cast<unsigned>(__builtin_ctz(m));
      Word& word = words[i];
      uint32_t v = word.load(std::memory_order_acquire);
      for (;;) {
        if (v == mine) break;
        if (v == 0 || v == tag) {
          if (!word.compare_exchange_weak(v, mine, std::memory_order_acq_rel)) continue;
          ++w.st.words_written_first;
          taken |= 1u << i;
          break;
        }
        // Another block's word, or one several blocks read. The words taken in this call are still unwritten: kept or not, undoing
        // leaves them as they are.
        if (v != kMany && (v & kWrite)) refuse("two blocks wrote the same bytes");
        // A reader's mark, which in a page that was first read by page state may say more than was read.
        if (pstate[slot].load(std::memory_order_acquire) == kWordsCoarse)
          refuse_coarse(base + (slot * kPageWords + l * kLineWords + i) * kGranule, "a block wrote what another block read");
        refuse("a block wrote what another block read");
      }
    }
    return taken;
  }

  // A write of words `m` of line `l` found the line another block's, or being written out (`v` is its record): stops the trial
  // where a word would, else sees the line noted by word. Returns the record as it is now.
  [[gnu::noinline]] uint64_t write_line_met(Worker& w, Page* page, uint64_t slot, uint64_t l, uint32_t m, uint64_t v) {
    Line& line = page->lines[l];
    const uint32_t o = owner(v);
    if (o == kLineBusy) {
      wait_busy(w);
      return line.load();
    }
    if (v == 0 || o == w.tag || o == kLineByWord) return v;   // (cannot be: the caller saw another block's)
    if (m & written_bits(v)) refuse("two blocks wrote the same bytes");
    if (const uint32_t r = m & read_bits(v)) {
      // A reader's mark, which in a page that was first read by page state may say more than was read.
      if (pstate[slot].load(std::memory_order_acquire) == kWordsCoarse)
        refuse_coarse(base + (slot * kPageWords + l * kLineWords + static_cast<unsigned>(__builtin_ctz(r))) * kGranule,
                      "a block wrote what another block read");
      refuse("a block wrote what another block read");
    }
    to_words(w, page, l, v);
    return v;
  }

  // A block is about to write in page `slot` (`addr` is the store's, for a coarse stop): the page comes to be noted by word, or the
  // trial stops. Leaves the page as the worker's last.
  [[gnu::noinline]] void write_page(Worker& w, uint64_t slot, uint64_t addr) {
    const uint32_t tag = w.tag;
    Word& ps = pstate[slot];
    uint32_t v = ps.load(std::memory_order_acquire);
    for (;;) {
      if (by_words(v)) break;
      if (v == 0) {
        if (!ps.compare_exchange_weak(v, kWords, std::memory_order_acq_rel)) continue;
        w.noted.push_back(slot);
        ++w.st.pages_by_word;
        break;
      }
      if (v == tag) {
        // The page's only reader so far. What it read was not noted by word, so every word starts as read by it (every line is
        // the block's, with all sixteen read); the page of lines is put in place before the state says there is one (no one
        // looks for it earlier).
        Page* mine = fresh_page(w);
        for (uint64_t i = 0; i < kPageLines; ++i) mine->lines[i].store(uint64_t{tag} | as_read(0xFFFFu));
        Page* none = nullptr;
        if (!top[slot].compare_exchange_strong(none, mine, std::memory_order_acq_rel)) {
          for (uint64_t i = 0; i < kPageLines; ++i) mine->lines[i].store(0);
          w.fresh.push_back(mine);
          refuse_coarse(addr, "a page read by page was found noted by word");   // (cannot happen: said, not assumed)
        }
        w.installed.push_back(slot);
        if (ps.compare_exchange_strong(v, kWordsCoarse, std::memory_order_acq_rel)) {
          ++w.st.pages_coarse;
          break;
        }
        // (another block read in the page meanwhile: v is kMany, and the page of words is never looked at)
      }
      refuse_coarse(addr, "a block wrote in a page other blocks read");
    }
    word_page(w, slot);
  }

  std::atomic<Page*>* top = nullptr;
  Word* pstate = nullptr;
  uint64_t top_slots = 0;
  uint64_t base = 0, span = 0;
  bool host_maps = false;
  bool plan_pages = true;
  std::vector<std::pair<uint64_t, uint64_t>> plan_words;
  std::atomic<bool> stopped{false};
  std::atomic<bool> coarse{false};
  std::atomic<const char*> why{nullptr};
  std::vector<Worker> workers;
  std::mutex pool_mu;
  std::vector<Page*> pool;   // zeroed pages
  std::vector<uint64_t> coarse_at;   // under pool_mu
};

namespace {
thread_local MemoryManager::Spec::Worker* t_spec_worker = nullptr;
}  // namespace

bool MemoryManager::spec_possible() const {
  if (memo_sink_ || memo_untrackable() || memo_faults_armed() || memo_trace_active()) return false;
  if (card_gate_.load(std::memory_order_relaxed) || card_resident_.load(std::memory_order_relaxed) > 0) return false;
  SharedGuard table_guard(table_lock_.get());
  for (const auto& [base, a] : live_)
    if (a.external) return false;   // bytes that live in a file another process may map
  return high_water_va_ > va_base_;
}

void MemoryManager::spec_begin(unsigned workers, const SpecPlan* plan) {
  if (!spec_kept_) spec_kept_ = std::make_shared<Spec>();
  Spec& s = *spec_kept_;
  s.size_for(va_base_, high_water_va_ - va_base_);
  s.host_maps = memo_host_maps();
  s.plan_pages = plan ? plan->pages : true;
  s.plan_words.clear();
  if (plan) s.plan_words = plan->by_word;
  s.stopped.store(false, std::memory_order_relaxed);
  s.coarse.store(false, std::memory_order_relaxed);
  s.why.store(nullptr, std::memory_order_relaxed);
  s.coarse_at.clear();
  if (s.workers.size() != workers) {
    s.workers.clear();
    s.workers.resize(workers);
  }
  for (Spec::Worker& w : s.workers) w.reset();
  spec_ = &s;
}

void MemoryManager::spec_enter(unsigned worker) { t_spec_worker = &spec_->workers[worker]; }

void MemoryManager::spec_block(uint64_t block) { t_spec_worker->tag = static_cast<uint32_t>(block + 1); }

void MemoryManager::spec_leave() { t_spec_worker = nullptr; }

bool MemoryManager::spec_stopped() const { return spec_ && spec_->stopped.load(std::memory_order_relaxed); }

void MemoryManager::spec_stop(const char* why) {
  if (!spec_) return;
  const char* none = nullptr;
  spec_->why.compare_exchange_strong(none, why, std::memory_order_relaxed);
  spec_->stopped.store(true, std::memory_order_release);
}

void MemoryManager::spec_refuse(const char* why) const { spec_->refuse(why); }

void MemoryManager::SpecStats::add(const SpecStats& o) {
  hooks += o.hooks;
  pages_read += o.pages_read;
  pages_shared += o.pages_shared;
  pages_by_word += o.pages_by_word;
  pages_coarse += o.pages_coarse;
  page_looks += o.page_looks;
  words_read += o.words_read;
  words_read_first += o.words_read_first;
  words_read_shared += o.words_read_shared;
  words_written += o.words_written;
  words_written_first += o.words_written_first;
  keeps += o.keeps;
  runs_known += o.runs_known;
  lines_by_word += o.lines_by_word;
  lines_one_block += o.lines_one_block;
}

MemoryManager::SpecOutcome MemoryManager::spec_end() {
  Spec& s = *spec_;
  spec_ = nullptr;   // (the workers have left: what follows is this thread's alone, and write() below is no trial's)
  SpecOutcome out;
  out.committed = !s.stopped.load(std::memory_order_acquire);
  if (const char* why = s.why.load(std::memory_order_relaxed)) out.why = why;
  for (Spec::Worker& w : s.workers) {
    out.kept_bytes += w.kept.size();
    out.shadow_pages += w.noted.size();
    out.stats.add(w.st);
    out.stats.word_pages += w.installed.size();
    if (!out.committed) {
      // Each piece is what the blocks of the trial found at bytes only this worker's block wrote, so the order of putting back is
      // free among them.
      uint64_t at = 0;
      for (size_t i = 0; i < w.kept_at.size(); ++i) {
        write(w.kept_at[i].first, w.kept.data() + at, w.kept_at[i].second);
        at += w.kept_at[i].second;
      }
    }
  }
  if (!out.committed && s.coarse.load(std::memory_order_acquire)) {
    // The allocations the coarse stops were in: what a second trial notes by word from the start.
    out.coarse = true;
    SharedGuard table_guard(table_lock_.get());
    for (const uint64_t addr : s.coarse_at) {
      auto up = live_.upper_bound(addr);
      if (up == live_.begin()) continue;
      const auto it = std::prev(up);
      const std::pair<uint64_t, uint64_t> range{it->first, it->first + it->second.size};
      if (addr < range.second && std::find(out.by_word.begin(), out.by_word.end(), range) == out.by_word.end()) out.by_word.push_back(range);
    }
  }
  // The shadow goes back to nothing: every page state is cleared, every installed page is zeroed and pooled, its slot cleared.
  std::lock_guard<std::mutex> g(s.pool_mu);
  for (Spec::Worker& w : s.workers) {
    for (const uint64_t slot : w.noted) s.pstate[slot].store(0, std::memory_order_relaxed);
    for (const uint64_t slot : w.installed) {
      Spec::Page* page = s.top[slot].exchange(nullptr, std::memory_order_acq_rel);
      // Zeroed again: the records, and the words of the lines that were written out (no other word was ever touched).
      for (uint64_t l = 0; l < Spec::kPageLines; ++l) {
        const uint64_t v = page->lines[l].load();
        if (v == 0) continue;
        if (Spec::owner(v) == Spec::kLineByWord) std::memset(static_cast<void*>(page->words + l * Spec::kLineWords), 0, Spec::kLineWords * sizeof(Spec::Word));
        else ++out.stats.lines_one_block;
      }
      std::memset(static_cast<void*>(page->lines), 0, sizeof(page->lines));
      if (s.pool.size() < Spec::kPoolPages) s.pool.push_back(page);
      else std::free(page);
    }
    for (Spec::Page* page : w.fresh) {   // zeroed and never installed
      if (s.pool.size() < Spec::kPoolPages) s.pool.push_back(page);
      else std::free(page);
    }
    w.fresh.clear();
    w.installed.clear();
    w.noted.clear();
    w.kept_at.clear();
    w.kept.clear();
    // (the log's memory stays for the next trial, unless it grew large)
    if (w.kept.capacity() > (size_t{64} << 20)) std::vector<uint8_t>().swap(w.kept);
    if (w.kept_at.capacity() > (size_t{1} << 20)) std::vector<std::pair<uint64_t, uint64_t>>().swap(w.kept_at);
  }
  s.coarse_at.clear();
  return out;
}

void MemoryManager::spec_keep(uint64_t addr, uint64_t len) const {
  Spec::Worker& w = *t_spec_worker;
  ++w.st.keeps;
  if (w.kept.size() + len > Spec::kKeepLimit) spec_refuse("the undo log passed its limit");
  // Wholly inside the allocation the last piece was in (the table cannot change during a launch: an allocation or a free stops the trial).
  if (addr >= w.keep_base && addr + len <= w.keep_end) {
    const size_t at = w.kept.size();
    w.kept.resize(at + len);
    read_chunks(*static_cast<const Allocation*>(w.keep_alloc), addr - w.keep_base, w.kept.data() + at, len);
    // (a piece that begins where the worker's last one ended is the same piece, longer: lines one after another)
    if (!w.kept_at.empty() && w.kept_at.back().first + w.kept_at.back().second == addr) w.kept_at.back().second += len;
    else w.kept_at.emplace_back(addr, len);
    return;
  }
  SharedGuard table_guard(table_lock_.get());
  const uint64_t end = addr + len;
  while (addr < end) {
    auto up = live_.upper_bound(addr);
    if (up == live_.begin() || addr >= std::prev(up)->first + std::prev(up)->second.size) {
      // No allocation holds this byte (padding after one, a freed range): nothing a store can change. On to the next allocation.
      if (up == live_.end() || up->first >= end) return;
      addr = up->first;
      continue;
    }
    const auto it = std::prev(up);
    w.keep_base = it->first;
    w.keep_end = it->first + it->second.size;
    w.keep_alloc = &it->second;
    const uint64_t n = std::min(end, w.keep_end) - addr;
    const size_t at = w.kept.size();
    w.kept.resize(at + n);
    read_chunks(it->second, addr - it->first, w.kept.data() + at, n);
    w.kept_at.emplace_back(addr, n);
    addr += n;
  }
}

// A block reads [addr, addr + len) (len > 0; the trial is on and this thread is `worker`).
[[gnu::always_inline]] inline void MemoryManager::spec_read_in(Spec& s, void* worker, uint64_t addr, uint64_t len) const {
  const uint64_t off = addr - s.base;
  // Outside the device's range: host memory (pinned, managed), which no block of a trial can write (spec_write_in), or
  // nothing at all, which the access itself reports.
  if (off >= s.span) return;
  Spec::Worker& w = *static_cast<Spec::Worker*>(worker);
  const uint64_t last = std::min(off + len, s.span) - 1;
  const uint64_t slot1 = last / Spec::kPageBytes;
  for (uint64_t slot = off / Spec::kPageBytes; slot <= slot1; ++slot) {
    // A page this block read by page state before, or that several blocks have: nothing more to note (no block can write in it).
    const Spec::Seen& seen = w.seen[Spec::seen_at(slot)];
    if (seen.slot == slot && (seen.tag == w.tag || seen.tag == Spec::kMany)) [[likely]] continue;
    const uint64_t lo = std::max(off, slot * Spec::kPageBytes), hi = std::min(last, slot * Spec::kPageBytes + Spec::kPageBytes - 1);
    const uint64_t g0 = lo / Spec::kGranule, g1 = hi / Spec::kGranule;
    // Words the block is remembered to have noted (a load in a loop).
    if (g1 - g0 + 1 >= Spec::kRunWorth && (Spec::known(w.wrote, w.tag, g0, g1) || Spec::known(w.read, w.tag, g0, g1))) {
      ++w.st.runs_known;
      continue;
    }
    // A page noted by word: by its lines. (The worker remembers where the lines of the pages it has read in are.)
    Spec::Worker::WordPage& wp = w.pages[Spec::seen_at(slot) % Spec::kWordPages];
    Spec::Page* page = wp.page;
    if (wp.slot != slot) {
      ++w.st.page_looks;
      page = s.read_page(w, slot);
      if (!page) continue;   // (noted by the page's state)
      wp = {slot, page};
    }
    w.st.words_read += g1 - g0 + 1;
    const uint64_t p0 = g0 % Spec::kPageWords, p1 = g1 % Spec::kPageWords;
    for (uint64_t l = p0 / Spec::kLineWords; l <= p1 / Spec::kLineWords; ++l) {
      const uint64_t first = std::max(p0, l * Spec::kLineWords), end = std::min(p1, l * Spec::kLineWords + Spec::kLineWords - 1);
      s.read_line(w, page, l, Spec::bits_of(first % Spec::kLineWords, end % Spec::kLineWords));
    }
    if (g1 - g0 + 1 >= Spec::kRunWorth) Spec::learn(w.read, w.read_next, w.tag, g0, g1);
  }
}

// A block is about to write [addr, addr + len): its pages come to be noted by word, it comes to own every word of it, and the
// bytes of the words it did not own yet are kept first (with the rest of their line, where the line is the block's alone).
[[gnu::always_inline]] inline void MemoryManager::spec_write_in(Spec& s, void* worker, uint64_t addr, uint64_t len) const {
  const uint64_t off = addr - s.base;
  // A store the shadow has no word for (host memory another thread or process may share; an address that is no
  // allocation's) cannot be undone or checked: the launch runs in order.
  if (off >= s.span || len > s.span - off) spec_refuse("a block wrote outside the device's own memory");
  if (s.host_maps && host_maps_ && host_maps_->may_overlap(addr, addr + len - 1)) spec_refuse("a block wrote host-mapped memory");
  Spec::Worker& w = *static_cast<Spec::Worker*>(worker);
  const uint64_t g0 = off / Spec::kGranule, g1 = (off + len - 1) / Spec::kGranule;
  // Words the block holds as their writer already (a store in a loop): nothing to note.
  if (g1 - g0 + 1 >= Spec::kRunWorth && Spec::known(w.wrote, w.tag, g0, g1)) {
    ++w.st.runs_known;
    return;
  }
  // The run of words taken just now and not kept yet: [run, run_end) in granules of the range.
  uint64_t run = ~uint64_t{0}, run_end = 0;
  const auto keep = [&] {
    if (run == ~uint64_t{0}) return;
    spec_keep(s.base + run * Spec::kGranule, (run_end - run) * Spec::kGranule);
    run = ~uint64_t{0};
  };
  w.st.words_written += g1 - g0 + 1;
  for (uint64_t g = g0; g <= g1;) {
    const uint64_t slot = g / Spec::kPageWords;
    // (A stop in write_page, or at a line or word below, leaves the words taken in this call unkept: they are still unwritten, so
    // undoing leaves them as they are.)
    if (slot != w.last_slot) [[unlikely]] {
      // (a page of lines the worker knows already: a warp's lanes are often each in a page of their own, the rows of a matrix)
      Spec::Worker::WordPage& wp = w.pages[Spec::seen_at(slot) % Spec::kWordPages];
      if (wp.slot == slot) {
        w.last_slot = slot;
        w.last_page = wp.page;
      } else {
        s.write_page(w, slot, s.base + g * Spec::kGranule);
        wp = {slot, w.last_page};
      }
    }
    Spec::Page* page = w.last_page;
    const uint64_t in_page = g % Spec::kPageWords, l = in_page / Spec::kLineWords;
    const uint64_t line0 = g - in_page % Spec::kLineWords;   // the granule of the line's first word
    const uint64_t last = std::min(g1, line0 + Spec::kLineWords - 1);
    const uint32_t m = Spec::bits_of(g - line0, last - line0);
    Spec::Line& line = page->lines[l];
    uint64_t v = line.load();
    uint32_t taken = 0;   // (words of a line noted by word that the block took just now: kept one by one, below)
    for (;;) {
      const uint32_t o = Spec::owner(v);
      if (o == w.tag && Spec::written_bits(v)) [[likely]] {
        // A line the block has written in before, alone: its 64 bytes are kept already.
        const uint32_t fresh = m & ~Spec::written_bits(v);
        if (fresh) VGPU_SPEC_HOOK(kOwnBefore);
        if (fresh && !line.take(v, v | Spec::as_written(fresh))) continue;   // (it moves under its block only when another writes it out)
        w.st.words_written_first += static_cast<uint64_t>(__builtin_popcount(fresh));
        break;
      }
      if (v == 0 || o == w.tag) {
        // The block's first write in a line nobody else has touched: the line's bytes are kept whole while the record is busy
        // (no block can come to the line meanwhile, so the copy is of bytes nobody is writing), then the record says what the
        // block wrote, and read.
        // (A stop from the log, its limit, leaves the record busy: the trial is over, and a block that waits at the line sees that.)
        if (!line.take(v, Spec::kLineBusy)) continue;
        VGPU_SPEC_HOOK(kKeepBusy);
        keep();
        spec_keep(s.base + line0 * Spec::kGranule, Spec::kLineWords * Spec::kGranule);
        line.store((v ? v : uint64_t{w.tag}) | Spec::as_written(m));
        w.st.words_written_first += static_cast<uint64_t>(__builtin_popcount(m));
        break;
      }
      if (o == Spec::kLineByWord) {
        taken = s.write_words(w, page, slot, l, m);
        break;
      }
      v = s.write_line_met(w, page, slot, l, m, v);
    }
    // The words taken are kept in runs (words next to one another, across lines and pages, are one piece of the log).
    while (taken) {
      const unsigned lo = static_cast<unsigned>(__builtin_ctz(taken));
      const unsigned n = static_cast<unsigned>(__builtin_ctz(~(taken >> lo)));
      if (run != ~uint64_t{0} && run_end == line0 + lo) {
        run_end += n;
      } else {
        keep();
        run = line0 + lo;
        run_end = run + n;
      }
      taken &= ~(((1u << n) - 1) << lo);
    }
    g = last + 1;
  }
  keep();
  if (g1 - g0 + 1 >= Spec::kRunWorth) Spec::learn(w.wrote, w.wrote_next, w.tag, g0, g1);
}

// The trial's worker of the calling thread, after the two tests every hook makes first.
[[gnu::always_inline]] inline void* MemoryManager::spec_worker() const {
  if (spec_->stopped.load(std::memory_order_relaxed)) throw SpecStop{};
  Spec::Worker* w = t_spec_worker;
  if (!w) spec_refuse("a thread outside the launch used device memory");
  ++w->st.hooks;
  return w;
}

void MemoryManager::spec_read(uint64_t addr, uint64_t len) const {
  void* w = spec_worker();
  if (len) spec_read_in(*spec_, w, addr, len);
}

void MemoryManager::spec_write(uint64_t addr, uint64_t len) const {
  void* w = spec_worker();
  if (len) spec_write_in(*spec_, w, addr, len);
}

void MemoryManager::spec_read_each(const uint64_t* addrs, size_t count, uint32_t size) const {
  void* w = spec_worker();
  Spec& s = *spec_;
  if (size)
    for (size_t i = 0; i < count; ++i) spec_read_in(s, w, addrs[i], size);
}

void MemoryManager::spec_write_each(const uint64_t* addrs, size_t count, uint32_t size) const {
  void* w = spec_worker();
  Spec& s = *spec_;
  if (size)
    for (size_t i = 0; i < count; ++i) spec_write_in(s, w, addrs[i], size);
}

void MemoryManager::spec_write_strided(uint64_t a0, int64_t stride, size_t count, uint32_t size) const {
  void* w = spec_worker();
  Spec& s = *spec_;
  if (size)
    for (size_t i = 0; i < count; ++i) spec_write_in(s, w, a0 + static_cast<uint64_t>(stride) * i, size);
}

}  // namespace vgpu
