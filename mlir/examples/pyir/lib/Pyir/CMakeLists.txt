add_mlir_dialect_library(MLIRPyir
        PyirDialect.cpp
        PyirOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Pyir

        DEPENDS
        MLIRPyirOpsIncGen
        )

target_link_libraries(MLIRPyir PUBLIC MLIRIR)
