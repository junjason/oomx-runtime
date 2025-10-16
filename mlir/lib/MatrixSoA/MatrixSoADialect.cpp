#include "mlir/IR/DialectImplementation.h"
using namespace mlir;
namespace mlir::msoa {
	struct MatrixDialect : public Dialect { MatrixDialect(MLIRContext* c): Dialect("msoa", c) {} };
}