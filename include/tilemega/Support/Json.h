// SPDX-License-Identifier: BSD-3-Clause
//
// TileMega -- Support/Json.h
//
// A minimal JSON value, parser and writer.
//
// It exists because configs/targets/*.json stopped being a flat table of
// scalars when Phase-4 calibration landed: the L2 bandwidth curve is an array,
// the atomic contention curve is an array, and the Stream-K coefficients are
// an array of objects keyed by tile shape.  The previous regex-per-key reader
// cannot express any of those, and a key-name regex silently reads the wrong
// occurrence once a name repeats inside a nested object.
#pragma once

#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace tilemega::json {

class Value;
using Object = std::vector<std::pair<std::string, Value>>;
using Array = std::vector<Value>;

class Value {
 public:
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject };

  Value() = default;
  Value(bool value) : kind_(Kind::kBool), boolean_(value) {}
  Value(double value) : kind_(Kind::kNumber), number_(value) {}
  Value(int value) : kind_(Kind::kNumber), number_(value) {}
  Value(char const* value) : kind_(Kind::kString), string_(value) {}
  Value(std::string value) : kind_(Kind::kString), string_(std::move(value)) {}
  Value(Array value) : kind_(Kind::kArray), array_(std::move(value)) {}
  Value(Object value) : kind_(Kind::kObject), object_(std::move(value)) {}

  Kind kind() const { return kind_; }
  bool IsNull() const { return kind_ == Kind::kNull; }

  /// Typed accessors.  Each throws std::runtime_error naming `context` when
  /// the value is absent or of the wrong kind -- a calibration file that lost
  /// a field must fail loudly, never read as zero.
  bool AsBool(char const* context) const;
  double AsNumber(char const* context) const;
  std::string const& AsString(char const* context) const;
  Array const& AsArray(char const* context) const;
  Object const& AsObject(char const* context) const;

  /// Object member lookup.  `At` throws when absent; `Find` returns nullptr.
  Value const& At(std::string const& key) const;
  Value const* Find(std::string const& key) const;

  /// Append to an object value (creating it when the value is null).
  void Set(std::string key, Value value);

  std::string Dump(int indent = 2) const;

 private:
  void Dump(std::string& out, int indent, int depth) const;

  Kind kind_ = Kind::kNull;
  bool boolean_ = false;
  double number_ = 0.0;
  std::string string_;
  Array array_;
  Object object_;
};

/// Parse `text`.  Throws std::runtime_error with a byte offset on malformed
/// input.
Value Parse(std::string const& text);

/// Parse the file at `path`.
Value ParseFile(std::string const& path);

/// Convenience builders, so callers read as JSON rather than as constructor
/// calls.
Value Numbers(std::vector<double> const& values);
Value Ints(std::vector<int> const& values);

}  // namespace tilemega::json
