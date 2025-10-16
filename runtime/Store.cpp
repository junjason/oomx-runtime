#include "oomx/Store.h"
#include <typeinfo>
#include <vector>
#include <cstdint>

namespace oomx {

Store::Store(const Schema& s, TileConfig) {
  for (auto& f : s.fields) {
    switch (f.type) {
      case FieldType::F32: cols_[f.name] = new Column<float>(); break;
      case FieldType::F64: cols_[f.name] = new Column<double>(); break;
      case FieldType::U32: cols_[f.name] = new Column<uint32_t>(); break;
      case FieldType::I32: cols_[f.name] = new Column<int32_t>(); break;
      case FieldType::BOOL: cols_[f.name] = new Column<uint8_t>(); break;
      default: cols_[f.name] = new Column<uint8_t>(); break;
    }
  }
}

Store::~Store() { for (auto& kv : cols_) delete kv.second; }

EntityId Store::create() {
  EntityId id = next_++;
  uint32_t row = count_++;
  id2row_[id] = row;
  for (auto& kv : cols_) {
    if (auto* cf = dynamic_cast<Column<float>*>(kv.second))        cf->data.emplace_back(0.f);
    else if (auto* cd = dynamic_cast<Column<double>*>(kv.second))   cd->data.emplace_back(0.0);
    else if (auto* cu = dynamic_cast<Column<uint32_t>*>(kv.second)) cu->data.emplace_back(0u);
    else if (auto* ci = dynamic_cast<Column<int32_t>*>(kv.second))  ci->data.emplace_back(0);
    else if (auto* cb = dynamic_cast<Column<uint8_t>*>(kv.second))  cb->data.emplace_back(0u);
  }
  return id;
}

void Store::destroy(EntityId) { /* free-list TBD */ }

uint32_t Store::rowOf(EntityId id) const { return id2row_.at(id); }

void Store::reorderRows(const std::vector<uint32_t>& newToOld) {
  const uint32_t N = count_;
  if (newToOld.size() != N || N == 0) return;

  // One reusable scratch buffer; we’ll reuse it for each column.
  // Size it large enough for the widest scalar we use here (double).
  std::vector<uint8_t> scratch;
  scratch.resize(static_cast<size_t>(N) * sizeof(double));

  auto permute = [&](auto* col) {
    using T = std::remove_pointer_t<decltype(col)>;
    const size_t bytes = static_cast<size_t>(N) * sizeof(T);
    if (scratch.size() < bytes) scratch.resize(bytes);
    T* tmp = reinterpret_cast<T*>(scratch.data());
    // Build new layout into tmp using permutation: newRow -> oldRow
    for (uint32_t nr = 0; nr < N; ++nr) {
      const uint32_t oldRow = newToOld[nr];
      tmp[nr] = col[oldRow];
    }
    // Copy back (tmp -> column)
    std::copy(tmp, tmp + N, col);
  };

  // Apply to each typed column we support.
  for (auto& kv : cols_) {
    if (auto* c = dynamic_cast<Column<float>*>(kv.second)) {
      permute(c->data.data());
    } else if (auto* c = dynamic_cast<Column<double>*>(kv.second)) {
      permute(c->data.data());
    } else if (auto* c = dynamic_cast<Column<uint32_t>*>(kv.second)) {
      permute(c->data.data());
    } else if (auto* c = dynamic_cast<Column<int32_t>*>(kv.second)) {
      permute(c->data.data());
    } else if (auto* c = dynamic_cast<Column<uint8_t>*>(kv.second)) {
      permute(c->data.data());
    }
  }

  // Rebuild id→row map (simple 1..N → 0..N-1 in MVP).
  // If you later keep stable external IDs that must follow rows, update here accordingly.
  uint32_t row = 0;
  for (auto& kv : id2row_) kv.second = row++;
}


// ---- explicit column<T> specializations ----
template<> float*     Store::column<float>(const char* n)    { return &static_cast<Column<float>*>(cols_.at(n))->data[0]; }
template<> double*    Store::column<double>(const char* n)   { return &static_cast<Column<double>*>(cols_.at(n))->data[0]; }
template<> uint32_t*  Store::column<uint32_t>(const char* n) { return &static_cast<Column<uint32_t>*>(cols_.at(n))->data[0]; }
template<> int32_t*   Store::column<int32_t>(const char* n)  { return &static_cast<Column<int32_t>*>(cols_.at(n))->data[0]; }
template<> uint8_t*   Store::column<uint8_t>(const char* n)  { return &static_cast<Column<uint8_t>*>(cols_.at(n))->data[0]; }

} // namespace oomx

