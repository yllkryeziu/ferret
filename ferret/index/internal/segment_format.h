// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The on-disk layout of a segment file (.fseg), shared by SegmentWriter and
// Segment and by nothing else.
//
//   [0, 64)              SegmentHeader
//   [64, 64 + 32 n)      SectionEntry[n], n = header.num_sections
//   16 KiB, 32 KiB, ...  sections, each starting at a multiple of
//                        kSectionAlignment (the arm64 macOS page size), so
//                        madvise can target one section
//   [size - 32, size)    SegmentFooter
//
// Sections, in the order the writer emits them:
//
//   kContent       the bytes of every document, concatenated in DocId
//                  order, then kContentPadding zero bytes
//   kChunks        ChunkRecord[num_chunks]
//   kDocs          DocRecord[num_docs]
//   kOids          20-byte Oid[num_docs]
//   kPaths         the hint paths, concatenated in DocId order
//   kPostings      the encoded posting lists (posting_list.h); a list of at
//                  least kPostingBlockSize ids starts at a multiple of
//                  kPostingPayloadAlignment
//   kTrigrams      TrigramEntry[num_trigrams], sorted by trigram
//   kTrigramIndex  u32[kTrigramIndexSize]: entry b is the index in kTrigrams
//                  of the first trigram whose top 16 bits are >= b
//
// Optional sections follow the paths: kSymbols and kCodeTokens.
//
// Everything is little-endian. Records are trivially copyable with fixed
// sizes, and are read with std::memcpy, so mapped bytes need no alignment.
// Readers ignore sections of unknown kinds, which lets later milestones add
// sections (kOidIndex, kExtSets, kSymbols, kCodeTokens) without a version bump.

#ifndef FERRET_INDEX_INTERNAL_SEGMENT_FORMAT_H_
#define FERRET_INDEX_INTERNAL_SEGMENT_FORMAT_H_

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"

namespace ferret::segment_format {

static_assert(std::endian::native == std::endian::little,
              "records are copied to and from disk as they are in memory");

inline constexpr char kHeaderMagic[8] = {'F', 'E', 'R', 'R',
                                         'E', 'T', 'S', 'G'};
inline constexpr char kFooterMagic[8] = {'F', 'S', 'E', 'G',
                                         'E', 'N', 'D', '\0'};
// Version 2 inserted RankTier::kDocumentation.
inline constexpr uint32_t kFormatVersion = 2;

inline constexpr uint64_t kSectionAlignment = uint64_t{16} * 1024;
// Zero bytes after the content, so scanners may read a little past a chunk.
inline constexpr uint64_t kContentPadding = 64;
inline constexpr size_t kTrigramIndexSize = (size_t{1} << 16) + 1;

// SegmentHeader::flags.
inline constexpr uint32_t kHeaderNoNewlineTrigrams = 1u << 0;
// Every line whose folded image is exactly two bytes "ab" contributes the
// pseudo-trigram "ab\n" to its chunk's postings (TrigramExtractor::
// ExtractShortLines). Segments without the flag, written by older builds or
// merged from such segments, have no such postings. Readers of older builds
// ignore the flag and the pseudo-trigrams alike.
inline constexpr uint32_t kHeaderShortLineTrigrams = 1u << 1;

struct SegmentHeader {
  char magic[8];
  uint32_t format_version;
  uint32_t flags;
  uint32_t chunk_bytes;
  uint32_t num_docs;
  uint32_t num_chunks;
  uint32_t num_trigrams;
  uint32_t num_sections;
  uint32_t reserved[6];
  // CRC32C of the 60 bytes before it.
  uint32_t crc32c;
};
static_assert(sizeof(SegmentHeader) == 64);

// The section table must fit in the first page, before the first section.
inline constexpr uint32_t kMaxSections =
    (kSectionAlignment - sizeof(SegmentHeader)) / 32;

// Persisted: never renumber.
enum class SectionKind : uint32_t {
  kContent = 1,
  kChunks = 2,
  kDocs = 3,
  kOids = 4,
  kPaths = 5,
  kPostings = 6,
  kTrigrams = 7,
  kTrigramIndex = 8,
  kSymbols = 9,
  kCodeTokens = 10,
};

struct SectionEntry {
  uint32_t kind;
  uint32_t flags;
  uint64_t offset;
  uint64_t length;
  // CRC32C of the section's bytes, checked by Segment::VerifyChecksums().
  uint32_t crc32c;
  uint32_t reserved;
};
static_assert(sizeof(SectionEntry) == 32);

struct ChunkRecord {
  // The chunk's bytes are content[content_offset, content_offset +
  // content_size) of the kContent section.
  uint64_t content_offset;
  uint32_t content_size;
  // 1-based.
  uint32_t first_line;
  uint32_t doc;
  // Bits 0-3: the LexState at the chunk's first byte.
  uint8_t flags;
  uint8_t pad[3];
};
static_assert(sizeof(ChunkRecord) == 24);

inline constexpr uint8_t kChunkLexStateMask = 0x0F;

struct DocRecord {
  uint32_t first_chunk;
  uint32_t num_chunks;
  uint32_t num_lines;
  // The hint path is paths[path_offset, next doc's path_offset), or up to
  // the end of kPaths for the last doc.
  uint32_t path_offset;
  uint16_t prior;
  uint8_t tier;
  uint8_t language;
  uint8_t flags;
  uint8_t pad[3];
};
static_assert(sizeof(DocRecord) == 24);

// DocRecord::flags.
inline constexpr uint8_t kDocLongLines = 1u << 0;
inline constexpr uint8_t kSymbolsPending = 1u << 1;
inline constexpr uint8_t kSymbolsFailed = 1u << 2;

// kSymbols starts with a pair of u32s (record count, pool size), followed by
// fixed records sorted by (folded name, name, doc, line), then a name pool.
struct SymbolRecord {
  uint32_t folded_offset;
  uint32_t name_offset;
  uint32_t doc;
  uint32_t line;
  uint32_t folded_size;
  uint32_t name_size;
  uint8_t kind;
  uint8_t flags;
  uint8_t pad[2];
};
static_assert(sizeof(SymbolRecord) == 28);
inline constexpr uint8_t kSymbolDefinition = 1u << 0;

// kCodeTokens: a header, then entries sorted by key, then the key pool, then
// the posting lists. A key is a folded code-identifier token
// (ferret/text/fold.h) of 1 to max_token_bytes bytes; its list holds the
// strictly increasing DocIds of the documents where some case variant occurs as
// a code identifier, each as a LEB128 varint: the first id, then the
// differences. Keys and lists are back to back in entry order, so an entry
// stores where its key and its list end; the start is the previous entry's end,
// and the last list ends where the section does.
struct CodeTokensHeader {
  uint32_t num_tokens;
  // Longer tokens are not stored, so a query for one cannot use the section.
  uint32_t max_token_bytes;
  uint32_t pool_bytes;
  uint32_t postings_bytes;
};
static_assert(sizeof(CodeTokensHeader) == 16);

struct CodeTokenEntry {
  // The offset in the key pool just past this key.
  uint32_t key_end;
  // The offset among the posting bytes just past this list, which is not
  // empty.
  uint32_t postings_end;
};
static_assert(sizeof(CodeTokenEntry) == 8);

// The longest folded token kCodeTokens stores.
inline constexpr uint32_t kMaxCodeTokenBytes = 64;

struct TrigramEntry {
  // Bits 0-23: the trigram. Bits 24-31: reserved, zero.
  uint32_t trigram_and_flags;
  // The number of ids in the posting list, at least 1.
  uint32_t count;
  // count <= kMaxInlinedPostings: the ids, the first in the low 32 bits and
  // the second, if any, in the high 32 bits. Otherwise: the offset of the
  // encoded list in the kPostings section.
  uint64_t postings;
};
static_assert(sizeof(TrigramEntry) == 16);

struct SegmentFooter {
  char magic[8];
  uint64_t file_length;
  // CRC32C of the section table.
  uint32_t section_table_crc32c;
  uint32_t reserved[3];
};
static_assert(sizeof(SegmentFooter) == 32);

// Appends the bytes of `record` to `out`.
template <typename Record>
void AppendRecord(const Record& record, std::string& out) {
  static_assert(std::is_trivially_copyable_v<Record>);
  out.append(reinterpret_cast<const char*>(&record), sizeof(Record));
}

// Reads the `index`-th Record of the array `bytes`.
// Requires (index + 1) * sizeof(Record) <= bytes.size() (DCHECKed).
template <typename Record>
Record ReadRecord(absl::string_view bytes, size_t index) {
  static_assert(std::is_trivially_copyable_v<Record>);
  DCHECK_LE((index + 1) * sizeof(Record), bytes.size());
  Record record;
  std::memcpy(&record, bytes.data() + index * sizeof(Record), sizeof(Record));
  return record;
}

}  // namespace ferret::segment_format

#endif  // FERRET_INDEX_INTERNAL_SEGMENT_FORMAT_H_
