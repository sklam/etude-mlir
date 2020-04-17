//===- PyirDialect.cpp - Pyir dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pyir/PyirDialect.h"
#include "Pyir/PyirOps.h"

using namespace mlir;
using namespace mlir::pyir;

//===----------------------------------------------------------------------===//
// Pyir dialect.
//===----------------------------------------------------------------------===//

PyirDialect::PyirDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "Pyir/PyirOps.cpp.inc"
      >();
}
