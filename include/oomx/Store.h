#pragma once
#include "Schema.h"
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace oomx {
using EntityId = uint32_t;
struct TileConfig { uint32_t tileRows=1024; uint32_t lane=32; bool aosoa=true; };

class Store {
public:
  explicit Store(const Schema& s, TileConfig cfg);
  ~Store();

  EntityId create();
  void destroy(EntityId id);
  uint32_t rowOf(EntityId id) const;

  template<typename T> T* column(const char* name); // explicit specs in .cpp
  uint32_t size() const { return count_; }

  // PUBLIC so Runtime can call it (dynamic SoA)
  void reorderRows(const std::vector<uint32_t>& newToOld);

  // Expose column node types so specializations can refer to them
  struct ColumnBase { virtual ~ColumnBase() = default; };
  template<typename T> struct Column : ColumnBase { std::vector<T> data; };

private:
  std::unordered_map<std::string, ColumnBase*> cols_;
  std::unordered_map<EntityId, uint32_t> id2row_;
  EntityId next_{1};
  uint32_t count_{0};
};

} // namespace oomx
