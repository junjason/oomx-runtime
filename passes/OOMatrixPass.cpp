#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"


using namespace llvm;


namespace {
struct OOMatrixPass : PassInfoMixin<OOMatrixPass> {
PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
for (Function &F : M) {
if (F.isDeclaration()) continue;
errs() << "[oomx] Analyze function: " << F.getName() << "\n";
// TODO: detect loops over arrays-of-structs, collect field GEPs, build FieldTable
// TODO: emit metadata or a sidecar plan for runtime to build SoA/AoSoA
}
return PreservedAnalyses::all();
}
};
} // anonymous


extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
return {LLVM_PLUGIN_API_VERSION, "oomx-pass", LLVM_VERSION_STRING,
[](PassBuilder &PB) {
PB.registerPipelineParsingCallback(
[](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
if (Name == "oomx-pass") { MPM.addPass(OOMatrixPass()); return true; }
return false;
});
}};
}