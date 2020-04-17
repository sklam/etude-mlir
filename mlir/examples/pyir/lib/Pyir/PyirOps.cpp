//===- PyirOps.cpp - Pyir dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pyir/PyirOps.h"
#include "Pyir/PyirDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace pyir {
#define GET_OP_CLASSES
#include "Pyir/PyirOps.cpp.inc"
} // namespace pyir
} // namespace mlir
