// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/manifest.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/crc/crc32c.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ferret {
namespace {

constexpr absl::string_view kMagic = "FERRETMF";
constexpr uint32_t kFormatVersion = 1;

void Put(uint64_t value, size_t width, std::string& out) {
  for (size_t i = 0; i < width; ++i) {
    out.push_back(static_cast<char>(value >> (8 * i)));
  }
}

absl::Status PutString(absl::string_view value, std::string& out) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError("MANIFEST string too long");
  }
  Put(value.size(), 4, out);
  out.append(value);
  return absl::OkStatus();
}

class Decoder {
 public:
  explicit Decoder(absl::string_view data) : data_(data) {}

  absl::StatusOr<uint64_t> Get(size_t width) {
    if (data_.size() - offset_ < width) {
      return absl::DataLossError("truncated MANIFEST integer");
    }
    uint64_t value = 0;
    for (size_t i = 0; i < width; ++i) {
      value |=
          static_cast<uint64_t>(static_cast<unsigned char>(data_[offset_ + i]))
          << (8 * i);
    }
    offset_ += width;
    return value;
  }

  absl::StatusOr<std::string> GetString() {
    auto length = Get(4);
    if (!length.ok()) return length.status();
    if (*length > data_.size() - offset_) {
      return absl::DataLossError("truncated MANIFEST string");
    }
    std::string value(data_.substr(offset_, *length));
    offset_ += *length;
    return value;
  }

  size_t remaining() const { return data_.size() - offset_; }

 private:
  absl::string_view data_;
  size_t offset_ = 0;
};

absl::Status Validate(const Manifest& manifest) {
  if (manifest.next_file_number == 0) {
    return absl::DataLossError("zero next file number");
  }
  absl::flat_hash_set<uint64_t> numbers;
  for (const ManifestSegment& segment : manifest.segments) {
    if (segment.file_number == 0 || segment.file_size == 0 ||
        segment.file_number >= manifest.next_file_number ||
        !numbers.insert(segment.file_number).second) {
      return absl::DataLossError("invalid or duplicate segment file number");
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> EncodeManifest(const Manifest& manifest) {
  absl::Status valid = Validate(manifest);
  if (!valid.ok()) return absl::InvalidArgumentError(valid.message());
  if (manifest.segments.size() > std::numeric_limits<uint32_t>::max() ||
      manifest.views.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError("MANIFEST count too large");
  }
  std::string out(kMagic);
  Put(kFormatVersion, 4, out);
  Put(manifest.manifest_seq, 8, out);
  Put(manifest.next_file_number, 8, out);
  Put(manifest.segments.size(), 4, out);
  for (const ManifestSegment& segment : manifest.segments) {
    Put(segment.file_number, 8, out);
    Put(segment.file_size, 8, out);
    Put(segment.num_docs, 4, out);
  }
  Put(manifest.views.size(), 4, out);
  for (const ManifestView& view : manifest.views) {
    valid = PutString(view.canonical_root, out);
    if (!valid.ok()) return valid;
    valid = PutString(view.git_common_dir, out);
    if (!valid.ok()) return valid;
  }
  Put(static_cast<uint32_t>(absl::ComputeCrc32c(out)), 4, out);
  return out;
}

absl::StatusOr<Manifest> DecodeManifest(absl::string_view bytes) {
  if (bytes.size() < kMagic.size() ||
      bytes.substr(0, kMagic.size()) != kMagic) {
    return absl::DataLossError("bad MANIFEST magic or truncated magic");
  }
  Decoder decoder(bytes.substr(kMagic.size()));
  auto version = decoder.Get(4);
  if (!version.ok()) return version.status();
  if (*version != kFormatVersion) {
    return absl::FailedPreconditionError(
        absl::StrCat("unknown MANIFEST format version ", *version));
  }
  if (bytes.size() < kMagic.size() + 4 + 4) {
    return absl::DataLossError("truncated MANIFEST CRC");
  }
  Decoder crc_reader(bytes.substr(bytes.size() - 4));
  auto crc = crc_reader.Get(4);
  if (*crc != static_cast<uint32_t>(
                  absl::ComputeCrc32c(bytes.substr(0, bytes.size() - 4)))) {
    return absl::DataLossError("bad MANIFEST CRC");
  }
  // Exclude the checksum so counts cannot consume it as payload.
  Decoder body(
      bytes.substr(kMagic.size() + 4, bytes.size() - kMagic.size() - 8));
  Manifest manifest;
  auto seq = body.Get(8);
  if (!seq.ok()) return seq.status();
  manifest.manifest_seq = *seq;
  auto next = body.Get(8);
  if (!next.ok()) return next.status();
  manifest.next_file_number = *next;
  auto count = body.Get(4);
  if (!count.ok()) return count.status();
  if (*count > body.remaining() / 20) {
    return absl::DataLossError("truncated MANIFEST segments");
  }
  for (uint64_t i = 0; i < *count; ++i) {
    auto number = body.Get(8);
    auto size = body.Get(8);
    auto docs = body.Get(4);
    manifest.segments.push_back({*number, *size, static_cast<uint32_t>(*docs)});
  }
  count = body.Get(4);
  if (!count.ok()) return count.status();
  if (*count > body.remaining() / 8) {
    return absl::DataLossError("truncated MANIFEST views");
  }
  for (uint64_t i = 0; i < *count; ++i) {
    auto root = body.GetString();
    if (!root.ok()) return root.status();
    auto git_dir = body.GetString();
    if (!git_dir.ok()) return git_dir.status();
    manifest.views.push_back({std::move(*root), std::move(*git_dir)});
  }
  if (body.remaining() != 0) {
    return absl::DataLossError("trailing MANIFEST bytes");
  }
  absl::Status valid = Validate(manifest);
  if (!valid.ok()) return valid;
  return manifest;
}

}  // namespace ferret
