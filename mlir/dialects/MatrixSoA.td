include "mlir/IR/OpBase.td"


def Matrix_Dialect : Dialect { let name = "msoa"; let cppNamespace = "mlir::msoa"; }


def MSoA_ColumnType : TypeDef<Matrix_Dialect, "Column"> { let summary="Column vector"; }


def MSoA_TileOp : Op<Matrix_Dialect, "tile", [NoSideEffect]> {
let summary = "Tile over SoA columns";
let arguments = (ins I64Attr:$tileRows, Variadic<AnyType>:$cols, AnyRegion:$body);
let results = (outs);
}


def MSoA_GatherOp : Op<Matrix_Dialect, "gather", [NoSideEffect]> {
let summary = "Gather from column by row index";
let arguments = (ins MSoA_ColumnType:$col, AnyType:$rowIdx);
let results = (outs AnyType:$val);
}


def MSoA_ScatterAddOp : Op<Matrix_Dialect, "scatter_add"> {
let summary = "Accumulate delta into column";
let arguments = (ins MSoA_ColumnType:$col, AnyType:$rowIdx, AnyType:$delta);
let results = (outs);
}