//===- PyirDialect.h - Pyir dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PYIR_PYIRDIALECT_H
#define PYIR_PYIRDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace pyir {

#include "Pyir/PyirOpsDialect.h.inc"

} // namespace pyir
} // namespace mlir

#endif // PYIR_PYIRDIALECT_H
