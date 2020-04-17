// RUN: pyir-opt --emit-llvm %s

/// Simply checking that we can indeed lower MLIR into LLVM
module {
    func @bar() -> i32 {
        %0 = constant 1 : i32
        return %0 : i32
    }
}
