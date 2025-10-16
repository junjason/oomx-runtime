#pragma once
#include <string>
#include <vector>
#include <cstdint>


namespace oomx {


enum class FieldType { F32, F64, I32, U32, I16, U16, I8, U8, BOOL };
struct FieldSpec { std::string name; FieldType type; uint32_t flags; };


struct Schema {
	std::vector<FieldSpec> fields;
	Schema& add(FieldSpec f) { fields.push_back(std::move(f)); return *this; }
	const FieldSpec* find(const std::string& n) const {
	for (auto& f: fields) if (f.name==n) return &f; return nullptr;
	}
};


} // namespace oomx