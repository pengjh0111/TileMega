// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Support/Json.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace tilemega::json {
namespace {

[[noreturn]] void Fail(std::string const& message) {
  throw std::runtime_error("json: " + message);
}

class Parser {
 public:
  explicit Parser(std::string const& text) : text_(text) {}

  Value ParseValue() {
    SkipSpace();
    if (at_ >= text_.size()) Fail("unexpected end of input");
    char c = text_[at_];
    switch (c) {
      case '{': return ParseObject();
      case '[': return ParseArray();
      case '"': return Value(ParseString());
      case 't': Expect("true"); return Value(true);
      case 'f': Expect("false"); return Value(false);
      case 'n': Expect("null"); return Value();
      default: return Value(ParseNumber());
    }
  }

  void Finish() {
    SkipSpace();
    if (at_ != text_.size()) Fail("trailing text at offset " + Where());
  }

 private:
  std::string Where() const { return std::to_string(at_); }

  void SkipSpace() {
    while (at_ < text_.size()) {
      char c = text_[at_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++at_;
      else break;
    }
  }

  void Expect(char const* literal) {
    for (char const* p = literal; *p; ++p) {
      if (at_ >= text_.size() || text_[at_] != *p)
        Fail(std::string("expected ") + literal + " at offset " + Where());
      ++at_;
    }
  }

  char Take(char expected) {
    SkipSpace();
    if (at_ >= text_.size() || text_[at_] != expected)
      Fail(std::string("expected '") + expected + "' at offset " + Where());
    return text_[at_++];
  }

  std::string ParseString() {
    Take('"');
    std::string out;
    while (true) {
      if (at_ >= text_.size()) Fail("unterminated string");
      char c = text_[at_++];
      if (c == '"') break;
      if (c != '\\') { out.push_back(c); continue; }
      if (at_ >= text_.size()) Fail("unterminated escape");
      char e = text_[at_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          // Only the BMP subset TileMega ever writes; a surrogate pair is not
          // silently mangled, it is rejected.
          if (at_ + 4 > text_.size()) Fail("truncated \\u escape");
          unsigned code = 0;
          for (int i = 0; i < 4; ++i) {
            char h = text_[at_++];
            code *= 16;
            if (h >= '0' && h <= '9') code += static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') code += static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') code += static_cast<unsigned>(h - 'A' + 10);
            else Fail("bad hex digit in \\u escape");
          }
          if (code >= 0xD800 && code <= 0xDFFF) Fail("surrogate pair unsupported");
          if (code < 0x80) {
            out.push_back(static_cast<char>(code));
          } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          } else {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          }
          break;
        }
        default: Fail("unknown escape");
      }
    }
    return out;
  }

  double ParseNumber() {
    std::size_t start = at_;
    if (at_ < text_.size() && (text_[at_] == '-' || text_[at_] == '+')) ++at_;
    while (at_ < text_.size()) {
      char c = text_[at_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
          c == '-' || c == '+') ++at_;
      else break;
    }
    if (start == at_) Fail("expected a value at offset " + std::to_string(start));
    try {
      return std::stod(text_.substr(start, at_ - start));
    } catch (std::exception const&) {
      Fail("malformed number at offset " + std::to_string(start));
    }
  }

  Value ParseArray() {
    Take('[');
    Array items;
    SkipSpace();
    if (at_ < text_.size() && text_[at_] == ']') { ++at_; return Value(items); }
    while (true) {
      items.push_back(ParseValue());
      SkipSpace();
      if (at_ < text_.size() && text_[at_] == ',') { ++at_; continue; }
      Take(']');
      break;
    }
    return Value(std::move(items));
  }

  Value ParseObject() {
    Take('{');
    Object members;
    SkipSpace();
    if (at_ < text_.size() && text_[at_] == '}') { ++at_; return Value(members); }
    while (true) {
      SkipSpace();
      std::string key = ParseString();
      Take(':');
      members.emplace_back(std::move(key), ParseValue());
      SkipSpace();
      if (at_ < text_.size() && text_[at_] == ',') { ++at_; continue; }
      Take('}');
      break;
    }
    return Value(std::move(members));
  }

  std::string const& text_;
  std::size_t at_ = 0;
};

void WriteString(std::string& out, std::string const& text) {
  out.push_back('"');
  for (char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                        static_cast<unsigned char>(c));
          out += buffer;
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

void WriteNumber(std::string& out, double value) {
  if (std::isfinite(value) && value == static_cast<double>(static_cast<long long>(value)) &&
      std::fabs(value) < 1e15) {
    out += std::to_string(static_cast<long long>(value));
    return;
  }
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%.10g", value);
  out += buffer;
}

}  // namespace

bool Value::AsBool(char const* context) const {
  if (kind_ != Kind::kBool) Fail(std::string(context) + " is not a boolean");
  return boolean_;
}

double Value::AsNumber(char const* context) const {
  if (kind_ != Kind::kNumber) Fail(std::string(context) + " is not a number");
  return number_;
}

std::string const& Value::AsString(char const* context) const {
  if (kind_ != Kind::kString) Fail(std::string(context) + " is not a string");
  return string_;
}

Array const& Value::AsArray(char const* context) const {
  if (kind_ != Kind::kArray) Fail(std::string(context) + " is not an array");
  return array_;
}

Object const& Value::AsObject(char const* context) const {
  if (kind_ != Kind::kObject) Fail(std::string(context) + " is not an object");
  return object_;
}

Value const* Value::Find(std::string const& key) const {
  if (kind_ != Kind::kObject) return nullptr;
  for (auto const& member : object_)
    if (member.first == key) return &member.second;
  return nullptr;
}

Value const& Value::At(std::string const& key) const {
  Value const* found = Find(key);
  if (found == nullptr) Fail("missing key \"" + key + "\"");
  return *found;
}

void Value::Set(std::string key, Value value) {
  if (kind_ == Kind::kNull) kind_ = Kind::kObject;
  if (kind_ != Kind::kObject) Fail("Set on a non-object value");
  for (auto& member : object_) {
    if (member.first == key) { member.second = std::move(value); return; }
  }
  object_.emplace_back(std::move(key), std::move(value));
}

void Value::Dump(std::string& out, int indent, int depth) const {
  std::string pad(static_cast<std::size_t>(indent * (depth + 1)), ' ');
  std::string close_pad(static_cast<std::size_t>(indent * depth), ' ');
  char const* newline = indent > 0 ? "\n" : "";
  switch (kind_) {
    case Kind::kNull: out += "null"; break;
    case Kind::kBool: out += boolean_ ? "true" : "false"; break;
    case Kind::kNumber: WriteNumber(out, number_); break;
    case Kind::kString: WriteString(out, string_); break;
    case Kind::kArray: {
      if (array_.empty()) { out += "[]"; break; }
      // A flat array of numbers stays on one line: the L2 and contention
      // curves are read by humans in diffs, and one point per line buries
      // them.
      bool flat = true;
      for (auto const& item : array_)
        if (item.kind_ != Kind::kNumber) { flat = false; break; }
      if (flat) {
        out += "[";
        for (std::size_t i = 0; i < array_.size(); ++i) {
          if (i) out += ", ";
          WriteNumber(out, array_[i].number_);
        }
        out += "]";
        break;
      }
      out += "["; out += newline;
      for (std::size_t i = 0; i < array_.size(); ++i) {
        out += pad;
        array_[i].Dump(out, indent, depth + 1);
        if (i + 1 < array_.size()) out += ",";
        out += newline;
      }
      out += close_pad; out += "]";
      break;
    }
    case Kind::kObject: {
      if (object_.empty()) { out += "{}"; break; }
      out += "{"; out += newline;
      for (std::size_t i = 0; i < object_.size(); ++i) {
        out += pad;
        WriteString(out, object_[i].first);
        out += ": ";
        object_[i].second.Dump(out, indent, depth + 1);
        if (i + 1 < object_.size()) out += ",";
        out += newline;
      }
      out += close_pad; out += "}";
      break;
    }
  }
}

std::string Value::Dump(int indent) const {
  std::string out;
  Dump(out, indent, 0);
  out.push_back('\n');
  return out;
}

Value Parse(std::string const& text) {
  Parser parser(text);
  Value value = parser.ParseValue();
  parser.Finish();
  return value;
}

Value ParseFile(std::string const& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open JSON file: " + path);
  std::string text{std::istreambuf_iterator<char>(input),
                   std::istreambuf_iterator<char>()};
  return Parse(text);
}

Value Numbers(std::vector<double> const& values) {
  Array items;
  items.reserve(values.size());
  for (double value : values) items.emplace_back(value);
  return Value(std::move(items));
}

Value Ints(std::vector<int> const& values) {
  Array items;
  items.reserve(values.size());
  for (int value : values) items.emplace_back(value);
  return Value(std::move(items));
}

}  // namespace tilemega::json
