#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include <iostream>
using namespace mlir;
namespace mlir::oomem {
struct OOMemDialect : public Dialect { OOMemDialect(MLIRContext* ctx): Dialect("oomem", ctx) {} };
} // namespace mlir::oomem