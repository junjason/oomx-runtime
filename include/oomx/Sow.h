#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace oomx {

class Sow {
public:
  using Row = uint32_t;
  struct FieldBufF32 { std::unordered_map<Row,float> add, set; };

  FieldBufF32& fieldF32(const std::string& name) { return f32_[name]; }

  std::vector<std::string> fieldNamesF32() const {
    std::vector<std::string> out; out.reserve(f32_.size());
    for (auto& kv : f32_) out.push_back(kv.first);
    return out;
  }

  bool empty() const { return f32_.empty(); }

private:
  std::unordered_map<std::string, FieldBufF32> f32_;
};

} // namespace oomx
