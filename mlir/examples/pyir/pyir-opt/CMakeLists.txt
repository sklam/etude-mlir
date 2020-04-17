get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRPyir
        )
add_llvm_executable(pyir-opt pyir-opt.cpp)

llvm_update_compile_flags(pyir-opt)
target_link_libraries(pyir-opt PRIVATE ${LIBS})
