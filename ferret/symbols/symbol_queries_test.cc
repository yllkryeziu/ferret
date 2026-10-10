// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Golden tests for queries/{c,cpp,python}.scm, run through SymbolExtractor.

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/symbols/symbol_extractor.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/symbols/testing/sample_sources.h"
#include "ferret/text/language.h"

namespace ferret {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

// Extracts the symbols of `source` and formats each one as
// "<line> <def|decl> <kind> <name>". One leading newline is dropped, so a raw
// string literal can start on the line after its opening delimiter and still
// have its first line numbered 1.
std::vector<std::string> Symbols(Language language, absl::string_view source) {
  SymbolExtractor extractor;
  const auto symbols =
      extractor.Extract(language, absl::StripPrefix(source, "\n"));
  std::vector<std::string> lines;
  if (!symbols.ok()) {
    ADD_FAILURE() << symbols.status();
    return lines;
  }
  for (const ExtractedSymbol& symbol : *symbols) {
    lines.push_back(
        absl::StrCat(symbol.line, symbol.is_definition ? " def " : " decl ",
                     SymbolKindName(symbol.kind), " ", symbol.name));
  }
  return lines;
}

std::vector<std::string> CSymbols(absl::string_view source) {
  return Symbols(Language::kC, source);
}

std::vector<std::string> CppSymbols(absl::string_view source) {
  return Symbols(Language::kCpp, source);
}

std::vector<std::string> PythonSymbols(absl::string_view source) {
  return Symbols(Language::kPython, source);
}

TEST(CQueryTest, LinuxKernelDefinitions) {
  // Excerpts from include/linux/slab.h, sched.h, list.h and
  // kernel/sched/core.c, respectively.
  EXPECT_THAT(CSymbols("void *kmalloc(size_t size, gfp_t flags);\n"
                       "#define kmalloc(size, flags) "
                       "alloc_hooks(kmalloc_noprof(size, flags))\n"),
              ElementsAre("1 decl function kmalloc", "2 def macro kmalloc"));
  EXPECT_THAT(CSymbols("struct task_struct {\n"
                       "  unsigned int __state;\n"
                       "  unsigned int saved_state;\n"
                       "};\n"),
              ElementsAre("1 def struct task_struct"));
  EXPECT_THAT(
      CSymbols("static __always_inline void list_add(struct list_head *new,\n"
               "                                     struct list_head *head)\n"
               "{\n"
               "  __list_add(new, head, head->next);\n"
               "}\n"),
      ElementsAre("1 def function list_add"));
  EXPECT_THAT(CSymbols("asmlinkage __visible void __sched schedule(void)\n"
                       "{\n"
                       "  struct task_struct *tsk = current;\n"
                       "  if (!task_is_running(tsk)) sched_submit_work(tsk);\n"
                       "}\n"),
              ElementsAre("1 def function schedule"));
}

TEST(CQueryTest, Sample) {
  EXPECT_THAT(
      CSymbols(ferret_testing::SampleSource(Language::kC)),
      ElementsAre(
          "7 def macro RING_MIN_SIZE", "8 def macro ring_is_empty",
          "12 def type_alias ring_cb", "14 def enum ring_state",
          "15 def enumerator RING_IDLE", "16 def enumerator RING_BUSY",
          "19 def struct ring", "30 def type_alias ring_t",
          "32 decl variable ring_debug", "33 def variable ring_count",
          "34 def variable ring_state_names", "36 decl function ring_init",
          "37 decl function ring_notify", "39 def function ring_init",
          "50 def function ring_notify", "60 def function ring_legacy_size",
          "67 def function ring_handler", "72 decl function ring_names"));
}

TEST(CQueryTest, PrototypesAreDeclarations) {
  EXPECT_THAT(
      CSymbols(R"c(
int open_file(const char *path);
static int *find_slot(int key);
char **split_words(const char *text), join_words(char **words);
int open_file(const char *path) { return 0; }
)c"),
      ElementsAre("1 decl function open_file", "2 decl function find_slot",
                  "3 decl function split_words", "3 decl function join_words",
                  "4 def function open_file"));
}

TEST(CQueryTest, KnrParameterDeclarationsAreNotVariables) {
  EXPECT_THAT(CSymbols(R"c(
int scale(value, factor)
    int value;
    double factor;
{
  return value * factor;
}
)c"),
              ElementsAre("1 def function scale"));
}

TEST(CQueryTest, FunctionPointerDeclaratorsDeclareObjects) {
  EXPECT_THAT(
      CSymbols(R"c(
void (*on_exit_hook)(int status);
int (*handlers[4])(void);
typedef int (*compare_fn)(const void *, const void *);
void (*install(int signo, void (*handler)(int)))(int);
struct ops { int (*read)(void); };
)c"),
      ElementsAre("1 def variable on_exit_hook", "2 def variable handlers",
                  "3 def type_alias compare_fn", "4 decl function install",
                  "5 def struct ops"));
}

TEST(CQueryTest, ExternVariablesAreDeclarations) {
  EXPECT_THAT(CSymbols(R"c(
extern int verbose;
extern void log_line(const char *line);
int verbose;
static int counter = 1;
int width, *cursor, grid[3][4];
)c"),
              ElementsAre("1 decl variable verbose", "2 decl function log_line",
                          "3 def variable verbose", "4 def variable counter",
                          "5 def variable width", "5 def variable cursor",
                          "5 def variable grid"));
}

TEST(CQueryTest, OnlySpecifiersWithABodyDefineTypes) {
  EXPECT_THAT(CSymbols(R"c(
struct list;
struct list *head;
struct list { struct list *next; };
union value { int i; double d; };
enum color { RED, GREEN = 2 };
typedef struct { int x, y; } point;
typedef struct node { struct node *next; } node_t, *node_p;
typedef unsigned int uint32_t;
)c"),
              ElementsAre("2 def variable head", "3 def struct list",
                          "4 def union value", "5 def enum color",
                          "5 def enumerator RED", "5 def enumerator GREEN",
                          "6 def type_alias point", "7 def struct node",
                          "7 def type_alias node_t", "7 def type_alias node_p",
                          "8 def type_alias uint32_t"));
}

TEST(CQueryTest, Macros) {
  EXPECT_THAT(CSymbols(R"c(
#define PAGE_SIZE 4096
#define ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#define HAVE_FEATURE
#ifndef MAX_DEPTH
#define MAX_DEPTH 8
#endif
)c"),
              ElementsAre("1 def macro PAGE_SIZE", "2 def macro ALIGN",
                          "3 def macro HAVE_FEATURE", "5 def macro MAX_DEPTH"));
}

TEST(CQueryTest, NamesInsideFunctionBodiesAreLocal) {
  // A macro defined in a function body is a helper for that function and is
  // conventionally #undef'd at its end.
  EXPECT_THAT(CSymbols(R"c(
int parse(const char *text)
{
  static int calls;
  struct token { int kind; } tok;
  typedef int offset_t;
  int helper(int);
#define SKIP(p) ((p) + 1)
#ifdef DEBUG
  int trace = 1;
#endif
  return tok.kind;
}
)c"),
              ElementsAre("1 def function parse"));
}

TEST(CQueryTest, FileScopePreprocessorBlocksAreNotLocal) {
  EXPECT_THAT(CSymbols(R"c(
#ifdef CONFIG_SMP
static int cpu_count;
int smp_init(void) { return 0; }
#else
int smp_init(void);
#endif
)c"),
              ElementsAre("2 def variable cpu_count", "3 def function smp_init",
                          "5 decl function smp_init"));
}

TEST(CQueryTest, LinuxAnnotationsDoNotHideFunctions) {
  EXPECT_THAT(CSymbols(R"c(
static int __init ring_module_init(void)
{
  return 0;
}
long copy_in(void __user *dst, const void *src);
)c"),
              ElementsAre("1 def function ring_module_init",
                          "5 decl function copy_in"));
}

TEST(CppQueryTest, Sample) {
  EXPECT_THAT(
      CppSymbols(ferret_testing::SampleSource(Language::kCpp)),
      ElementsAre("10 def type_alias EntryId", "12 def enum CacheType",
                  "13 def enumerator kDisk", "14 def enumerator kMemory",
                  "17 def variable kMaxEntries",
                  "18 decl variable kCacheDirName", "21 def class ScopedRef",
                  "23 def method ScopedRef", "24 decl method ~ScopedRef",
                  "26 def method get", "27 decl method operator*",
                  "33 def class DiskCache", "35 def struct Options",
                  "39 decl method Create", "41 def method DiskCache",
                  "42 def method operator=", "43 decl method ~DiskCache",
                  "45 decl method OpenEntry", "46 decl method Flush",
                  "47 decl method operator bool", "49 decl function operator==",
                  "52 decl method DiskCache", "59 def variable kIndexFile",
                  "61 def struct EntryRecord", "66 def function Clamp",
                  "70 def variable kCacheDirName", "73 def method ~ScopedRef",
                  "76 def method operator*", "80 def method DiskCache",
                  "81 def method ~DiskCache", "84 def method Create",
                  "90 def method OpenEntry", "98 def method operator bool",
                  "100 def function operator=="));
}

TEST(CppQueryTest, PrototypesAreDeclarations) {
  EXPECT_THAT(
      CppSymbols(R"cpp_src(
int Parse(std::string_view text);
[[nodiscard]] static const char* Describe(int code) noexcept;
auto Trailing() -> int;
std::string MostVexing();
int Parse(std::string_view text) { return 0; }
auto Trailing() -> int { return 1; }
)cpp_src"),
      ElementsAre("1 decl function Parse", "2 decl function Describe",
                  "3 decl function Trailing", "4 decl function MostVexing",
                  "5 def function Parse", "6 def function Trailing"));
}

TEST(CppQueryTest, InClassMemberFunctionDeclarationsAreDeclarations) {
  EXPECT_THAT(CppSymbols(R"cpp_src(
class Widget : public View {
 public:
  Widget();
  explicit Widget(int id) : id_(id) {}
  Widget(const Widget&) = delete;
  Widget& operator=(Widget&&) = default;
  ~Widget() override;
  virtual void Paint(Canvas& canvas) const = 0;
  static Widget* Create(int id);
  int id() const { return id_; }
  bool operator<(const Widget& other) const;
  explicit operator bool() const;
};
)cpp_src"),
              ElementsAre("1 def class Widget", "3 decl method Widget",
                          "4 def method Widget", "5 def method Widget",
                          "6 def method operator=", "7 decl method ~Widget",
                          "8 decl method Paint", "9 decl method Create",
                          "10 def method id", "11 decl method operator<",
                          "12 decl method operator bool"));
}

TEST(CppQueryTest, OutOfLineMethodsAreDefinitions) {
  EXPECT_THAT(CppSymbols(R"cpp_src(
Widget::Widget() = default;
Widget::~Widget() {}
void Widget::Paint(Canvas& canvas) const {}
Widget* Widget::Create(int id) { return nullptr; }
const char** Widget::Names() { return nullptr; }
Widget& Widget::operator=(const Widget& other) { return *this; }
Widget::operator bool() const { return true; }
void* Widget::operator new(size_t size) { return nullptr; }
void ui::Widget::Layout::Run() {}
)cpp_src"),
              ElementsAre("1 def method Widget", "2 def method ~Widget",
                          "3 def method Paint", "4 def method Create",
                          "5 def method Names", "6 def method operator=",
                          "7 def method operator bool",
                          "8 def method operator new", "9 def method Run"));
}

TEST(CppQueryTest, NamespaceQualifiedFreeFunctionReadsAsMethod) {
  // `ns::Free` and `Class::Method` are the same syntax; without semantic
  // analysis a qualified definition is taken to be a method.
  EXPECT_THAT(CppSymbols(R"cpp_src(
namespace ns {
void Free();
}
void ns::Free() {}
)cpp_src"),
              ElementsAre("2 decl function Free", "4 def method Free"));
}

TEST(CppQueryTest, Templates) {
  EXPECT_THAT(CppSymbols(R"cpp_src(
template <typename T>
T Max(T a, T b) { return a < b ? b : a; }
template <typename T> void Swap(T& a, T& b);
template <> void Swap<int>(int& a, int& b) {}
template <typename T>
class Vec {
 public:
  template <typename U> void Emplace(U&& value);
  void Push(T value);
};
template <typename T>
void Vec<T>::Push(T value) {}
template <> class Vec<bool> {};
template <typename T> using Ptr = T*;
template <typename T> constexpr bool kIsVec = false;
)cpp_src"),
              ElementsAre("2 def function Max", "3 decl function Swap",
                          "4 def function Swap", "6 def class Vec",
                          "8 decl method Emplace", "9 decl method Push",
                          "12 def method Push", "13 def class Vec",
                          "14 def type_alias Ptr", "15 def variable kIsVec"));
}

TEST(CppQueryTest, NestedNamespacesAreTransparent) {
  EXPECT_THAT(CppSymbols(R"cpp_src(
namespace chrome::browser {
namespace {
constexpr int kLimit = 3;
}  // namespace
inline namespace v2 {
void Start();
}  // namespace v2
namespace internal {
struct State {};
}  // namespace internal
}  // namespace chrome::browser
)cpp_src"),
              ElementsAre("3 def variable kLimit", "6 decl function Start",
                          "9 def struct State"));
}

TEST(CppQueryTest, OperatorNamesAreNormalized) {
  EXPECT_THAT(
      CppSymbols(R"cpp_src(
struct Money {
  Money operator + (const Money& other) const;
  bool operator ==(const Money& other) const;
  void* operator new [] (size_t size);
  int operator()(int x) const;
  operator const char *() const;
};
Money operator"" _usd(unsigned long long cents);
std::ostream& operator<<(std::ostream& out, const Money& money);
)cpp_src"),
      ElementsAre(
          "1 def struct Money", "2 decl method operator+",
          "3 decl method operator==", "4 decl method operator new[]",
          "5 decl method operator()", "6 decl method operator const char*",
          "8 decl function operator\"\"_usd", "9 decl function operator<<"));
}

TEST(CppQueryTest, ExportMacrosDoNotHideClasses) {
  EXPECT_THAT(CppSymbols(R"cpp_src(
class CONTENT_EXPORT WebContents : public PageNavigator {
 public:
  static std::unique_ptr<WebContents> Create();
  ~WebContents() override;
};
struct COMPONENT_EXPORT(NETWORK) Request final {
  void Start();
};
class ABSL_MUST_USE_RESULT BASE_EXPORT Status {};
)cpp_src"),
              ElementsAre("1 def class WebContents", "3 decl method Create",
                          "4 decl method ~WebContents", "6 def struct Request",
                          "7 decl method Start", "9 def class Status"));
}

TEST(CppQueryTest, GnuVisibilityAttributeDoesNotHideClass) {
  EXPECT_THAT(
      CppSymbols("class __attribute__((visibility(\"default\"))) Api {\n"
                 "  void Run();\n"
                 "};\n"),
      ElementsAre("1 def class Api", "2 decl method Run"));
}

TEST(CppQueryTest, FriendsAreFreeFunctions) {
  EXPECT_THAT(CppSymbols(R"cpp_src(
class Point {
  friend class Canvas;
  friend void Swap(Point& a, Point& b);
  friend bool operator==(const Point& a, const Point& b) { return true; }
  int x_;
};
)cpp_src"),
              ElementsAre("1 def class Point", "3 decl function Swap",
                          "4 def function operator=="));
}

TEST(CppQueryTest, DataMembersAreNotReportedButNestedTypesAre) {
  EXPECT_THAT(
      CppSymbols(R"cpp_src(
class Cache {
 public:
  using Key = std::string;
  typedef int Size;
  enum class Mode { kRead, kWrite };
  struct Entry { int hits; };
  static constexpr int kMaxEntries = 64;
  static inline int instances = 0;
 private:
  int size_ = 0;
  void (*on_evict_)(int key);
};
int Cache::instances_created = 0;
)cpp_src"),
      ElementsAre("1 def class Cache", "3 def type_alias Key",
                  "4 def type_alias Size", "5 def enum Mode",
                  "5 def enumerator kRead", "5 def enumerator kWrite",
                  "6 def struct Entry", "13 def variable instances_created"));
}

TEST(CppQueryTest, MembersInPreprocessorBlocksAreMethods) {
  EXPECT_THAT(CppSymbols(R"cpp_src(
class Window {
#if BUILDFLAG(IS_WIN)
  explicit Window(HWND hwnd);
  HWND hwnd() const { return hwnd_; }
#endif
};
)cpp_src"),
              ElementsAre("1 def class Window", "3 decl method Window",
                          "4 def method hwnd"));
}

TEST(CppQueryTest, NamesInsideFunctionBodiesAndLambdasAreLocal) {
  EXPECT_THAT(CppSymbols(R"cpp_src(
void Run() {
  struct Visitor {
    void Visit() {}
  };
  std::string name();
  int count = 0;
  auto log = [](int level) { int depth = level; return depth; };
}
auto kCompare = [](int a, int b) { struct Tmp {}; return a < b; };
)cpp_src"),
              ElementsAre("1 def function Run", "9 def variable kCompare"));
}

TEST(CppQueryTest, ExternAndLinkageSpecifications) {
  EXPECT_THAT(
      CppSymbols(R"cpp_src(
extern const char kVersion[];
extern "C" int legacy_flag;
extern "C" {
int shared_counter;
void shared_reset(void);
}
)cpp_src"),
      ElementsAre("1 decl variable kVersion", "2 decl variable legacy_flag",
                  "4 def variable shared_counter",
                  "5 decl function shared_reset"));
}

TEST(CppQueryTest, ClassKinds) {
  EXPECT_THAT(CppSymbols(R"cpp_src(
class Forward;
enum class Opaque : int;
class Outer::Impl {};
union Bits { int i; float f; };
enum Plain : int { kFirst };
)cpp_src"),
              ElementsAre("3 def class Impl", "4 def union Bits",
                          "5 def enum Plain", "5 def enumerator kFirst"));
}

TEST(CppQueryTest, Macros) {
  EXPECT_THAT(
      CppSymbols(R"cpp_src(
#define DISALLOW_COPY(T) T(const T&) = delete
#define BASE_EXPORT __attribute__((visibility("default")))
)cpp_src"),
      ElementsAre("1 def macro DISALLOW_COPY", "2 def macro BASE_EXPORT"));
}

TEST(PythonQueryTest, Sample) {
  EXPECT_THAT(
      PythonSymbols(ferret_testing::SampleSource(Language::kPython)),
      ElementsAre("10 def variable T", "11 def variable logger",
                  "12 def variable DEFAULT_ATTEMPTS",
                  "13 decl variable _registry", "15 def type_alias Predicate",
                  "18 def class RetryError", "21 def method __init__",
                  "27 def function retry", "45 def class Backoff",
                  "48 def class Policy", "49 def method next_delay",
                  "53 def method default", "56 def method sleep",
                  "60 def method jitter"));
}

TEST(PythonQueryTest, FunctionsAndClasses) {
  EXPECT_THAT(PythonSymbols(R"py(
def load(path):
    return open(path).read()

async def fetch(url):
    return await client.get(url)

class Config:
    pass
)py"),
              ElementsAre("1 def function load", "4 def function fetch",
                          "7 def class Config"));
}

TEST(PythonQueryTest, MethodsIncludeDecoratedAndAsyncOnes) {
  EXPECT_THAT(PythonSymbols(R"py(
class Client:
    timeout = 30

    def __init__(self, host):
        self.host = host

    @property
    def url(self):
        return self.host

    @staticmethod
    @functools.cache
    async def resolve(name):
        pass
)py"),
              ElementsAre("1 def class Client", "4 def method __init__",
                          "8 def method url", "13 def method resolve"));
}

TEST(PythonQueryTest, NestedClassesAndTheirMethods) {
  EXPECT_THAT(PythonSymbols(R"py(
class Outer:
    class Inner:
        class Innermost:
            def deep(self):
                pass

        def shallow(self):
            pass
)py"),
              ElementsAre("1 def class Outer", "2 def class Inner",
                          "3 def class Innermost", "4 def method deep",
                          "7 def method shallow"));
}

TEST(PythonQueryTest, NamesInsideFunctionsAreLocal) {
  EXPECT_THAT(PythonSymbols(R"py(
def retry(times):
    def decorator(fn):
        def wrapper(*args):
            return fn(*args)
        return wrapper
    class Attempt:
        def run(self):
            pass
    return decorator
)py"),
              ElementsAre("1 def function retry"));
}

TEST(PythonQueryTest, ModuleLevelAssignments) {
  EXPECT_THAT(
      PythonSymbols(R"py(
MAX_RETRIES = 3
timeout: float = 1.5
cache: dict
first = second = None
x, y = 0, 0
settings.debug = True

if __name__ == "__main__":
    main_result = run()
try:
    import ujson as json
except ImportError:
    json = None
)py"),
      ElementsAre("1 def variable MAX_RETRIES", "2 def variable timeout",
                  "3 decl variable cache", "4 def variable first"));
}

TEST(PythonQueryTest, ConditionalDefinitions) {
  EXPECT_THAT(PythonSymbols(R"py(
if sys.platform == "win32":
    def home():
        return os.environ["USERPROFILE"]
else:
    def home():
        return os.environ["HOME"]

class Path:
    if sys.version_info >= (3, 12):
        def walk(self):
            pass
)py"),
              ElementsAre("2 def function home", "5 def function home",
                          "8 def class Path", "10 def method walk"));
}

TEST(PythonQueryTest, TypeAliasStatements) {
  EXPECT_THAT(PythonSymbols(R"py(
type Vector = list[float]
type Pair[T] = tuple[T, T]
)py"),
              ElementsAre("1 def type_alias Vector", "2 def type_alias Pair"));
}

TEST(SymbolQueriesTest, EmptyFilesHaveNoSymbols) {
  EXPECT_THAT(CSymbols(""), IsEmpty());
  EXPECT_THAT(CppSymbols(""), IsEmpty());
  EXPECT_THAT(PythonSymbols(""), IsEmpty());
}

}  // namespace
}  // namespace ferret
