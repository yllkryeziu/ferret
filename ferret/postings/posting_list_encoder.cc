// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/postings/posting_list_encoder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#include "absl/log/check.h"
#include "absl/types/span.h"

#include "ferret/postings/bitpack.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/internal/posting_list_format.h"
#include "ferret/postings/posting_list.h"
#include "ferret/postings/varint.h"

namespace ferret {

using postings_internal::kBeforeFirstId;
using postings_internal::PackOffset16AndWidth;
using postings_internal::PayloadRegionOffset;
using postings_internal::SkipEntry;

void AppendEncodedPostingList(absl::Span<const ChunkId> ids, std::string& out) {
  CHECK(!IsInlinedPostingList(ids.size()))
      << "a list of " << ids.size() << " ids belongs in its dictionary entry";
  CHECK_LE(ids.size(), kMaxPostingListSize);
  DCHECK(std::adjacent_find(ids.begin(), ids.end(),
                            std::greater_equal<ChunkId>()) == ids.end())
      << "ids are not strictly increasing";

  const size_t start = out.size();
  AppendVarint32(static_cast<uint32_t>(ids.size()), out);
  const size_t num_full_blocks = ids.size() / kPostingBlockSize;
  ChunkId previous = kBeforeFirstId;
  if (num_full_blocks > 0) {
    const auto payload_offset = static_cast<size_t>(
        PayloadRegionOffset(static_cast<int64_t>(out.size() - start),
                            static_cast<int64_t>(num_full_blocks)));
    const size_t skip_table =
        start + payload_offset - num_full_blocks * sizeof(SkipEntry);
    // Zero padding, then room for the skip table.
    out.resize(start + payload_offset, '\0');
    uint32_t offset16 = 0;
    for (size_t block = 0; block < num_full_blocks; ++block) {
      const absl::Span<const ChunkId> block_ids =
          ids.subspan(block * kPostingBlockSize, kPostingBlockSize);
      const int bit_width =
          DeltaBlockBitWidth(DeltaCoding::kD1, previous, block_ids);
      const SkipEntry entry = {
          .last_chunk = block_ids.back(),
          .offset16_and_width = PackOffset16AndWidth(offset16, bit_width)};
      std::memcpy(out.data() + skip_table + block * sizeof(SkipEntry), &entry,
                  sizeof(entry));
      const size_t payload = out.size();
      out.resize(payload + static_cast<size_t>(PackedBlockBytes(bit_width)));
      PackDeltaBlock(DeltaCoding::kD1, previous, block_ids, bit_width,
                     reinterpret_cast<uint8_t*>(out.data() + payload));
      offset16 += static_cast<uint32_t>(bit_width);
      previous = block_ids.back();
    }
  }
  for (const ChunkId id : ids.subspan(num_full_blocks * kPostingBlockSize)) {
    AppendVarint32(id - previous - 1, out);
    previous = id;
  }
}

}  // namespace ferret
