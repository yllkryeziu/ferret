"""Compiler flags shared by every first-party target.

Pass FERRET_DEFAULT_COPTS to the `copts` of each cc_library/cc_binary and
FERRET_TEST_COPTS to each cc_test, in the style of ABSL_DEFAULT_COPTS.
Packages that do byte-level arithmetic (//ferret/text, //ferret/regex) add
FERRET_STRICT_SIGN_COPTS: on arm64 `char` is signed, and a silent sign
extension while packing trigrams would be a false negative, not a crash.
"""

FERRET_DEFAULT_COPTS = [
    "-fno-exceptions",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wimplicit-fallthrough",
    "-Wmissing-declarations",
    "-Wnon-virtual-dtor",
    "-Woverloaded-virtual",
    "-Wshadow",
    "-Wshorten-64-to-32",
    "-Wthread-safety",
    "-Wunreachable-code",
    "-Wvla",
]

FERRET_TEST_COPTS = FERRET_DEFAULT_COPTS

FERRET_STRICT_SIGN_COPTS = ["-Wsign-conversion"]
