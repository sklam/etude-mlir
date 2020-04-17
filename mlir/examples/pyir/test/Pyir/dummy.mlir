// RUN: pyir-opt %s | pyir-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = pyir.foo %{{.*}} : i32
        %res = pyir.foo %0 : i32
        return
    }
}
