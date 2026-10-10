"""Module extension that fetches the tree-sitter runtime and pinned grammars."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_RUNTIME = struct(
    version = "0.27.0",
    sha256 = "d35c96e68736bd9569d2757c3cc71052485f33082c3825f1aed9d0e86013a159",
)

# name -> (version, sha256, source dirs relative to the archive root).
# Every source dir must contain parser.c and may contain scanner.c.
_GRAMMARS = {
    "c": ("0.24.2", "2eeb4db31f8fa0865e45488503d13403923bcb485a1bdb637abff8c42dd97364", ["src"]),
    "cpp": ("0.23.4", "7a2c55afe3028f4105f25762ea58cc16537d1f5a1dcd9cca90410b3cd5d46051", ["src"]),
    "python": ("0.25.0", "4609a3665a620e117acf795ff01b9e965880f81745f287a16336f4ca86cf270c", ["src"]),
}

_RUNTIME_BUILD = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "tree_sitter",
    srcs = ["lib/src/lib.c"],
    hdrs = ["lib/include/tree_sitter/api.h"],
    copts = ["-w", "-std=c11", "-D_DEFAULT_SOURCE", "-D_DARWIN_C_SOURCE"],
    includes = ["lib/include"],
    textual_hdrs = glob(["lib/src/**/*.h", "lib/src/**/*.c"], exclude = ["lib/src/lib.c"]),
    visibility = ["//visibility:public"],
)
"""

def _grammar_build(src_dirs):
    return """
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "grammar",
    srcs = glob({srcs}, allow_empty = True),
    copts = ["-w", "-std=c11"],
    includes = {includes},
    textual_hdrs = glob({hdrs}, allow_empty = True),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "tags_query",
    srcs = glob(["queries/tags.scm"], allow_empty = True),
    visibility = ["//visibility:public"],
)
""".format(
        srcs = repr([d + "/parser.c" for d in src_dirs] + [d + "/scanner.c" for d in src_dirs]),
        includes = repr(src_dirs),
        hdrs = repr([d + "/tree_sitter/*.h" for d in src_dirs]),
    )

def _tree_sitter_impl(module_ctx):
    http_archive(
        name = "tree_sitter",
        build_file_content = _RUNTIME_BUILD,
        sha256 = _RUNTIME.sha256,
        strip_prefix = "tree-sitter-" + _RUNTIME.version,
        urls = ["https://github.com/tree-sitter/tree-sitter/archive/refs/tags/v{}.tar.gz".format(_RUNTIME.version)],
    )
    for lang, (version, sha256, src_dirs) in _GRAMMARS.items():
        http_archive(
            name = "tree_sitter_" + lang,
            build_file_content = _grammar_build(src_dirs),
            sha256 = sha256,
            strip_prefix = "tree-sitter-{}-{}".format(lang, version),
            urls = ["https://github.com/tree-sitter/tree-sitter-{}/archive/refs/tags/v{}.tar.gz".format(lang, version)],
        )
    return module_ctx.extension_metadata(reproducible = True)

tree_sitter = module_extension(implementation = _tree_sitter_impl)
