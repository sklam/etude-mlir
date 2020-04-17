//===- PyirDialect.td - Pyir dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PYIR_DIALECT
#define PYIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Pyir dialect definition.
//===----------------------------------------------------------------------===//

def Pyir_Dialect : Dialect {
    let name = "pyir";
    let summary = "A pyir out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "pyir";
}

//===----------------------------------------------------------------------===//
// Base pyir operation definition.
//===----------------------------------------------------------------------===//

class Pyir_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Pyir_Dialect, mnemonic, traits>;

#endif // PYIR_DIALECT
