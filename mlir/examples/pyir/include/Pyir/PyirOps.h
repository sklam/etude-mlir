//===- PyirOps.h - Pyir dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PYIR_PYIROPS_H
#define PYIR_PYIROPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffects.h"

namespace mlir {
namespace pyir {

#define GET_OP_CLASSES
#include "Pyir/PyirOps.h.inc"

} // namespace pyir
} // namespace mlir

#endif // PYIR_PYIROPS_H
