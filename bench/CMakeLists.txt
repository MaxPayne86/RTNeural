include_directories(../RTNeural)

# set optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=cortex-a8 -mtune=cortex-a8 -mfloat-abi=hard -mfpu=neon -ftree-vectorize -ffast-math -fprefetch-loop-arrays -funroll-loops -funsafe-loop-optimizations -fno-finite-math-only")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-Wl,-Ofast -Wl,--as-needed -Wl,--strip-debug")
message("CMAKE_CXX_FLAGS_RELEASE in ${CMAKE_CURRENT_SOURCE_DIR} = ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_SHARED_LINKER_FLAGS_RELEASE in ${CMAKE_CURRENT_SOURCE_DIR} = ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")

add_executable(rtneural_layer_bench layer_bench.cpp)
target_link_libraries(rtneural_layer_bench LINK_PUBLIC RTNeural)

add_custom_command(TARGET rtneural_layer_bench
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:rtneural_layer_bench> to ${PROJECT_BINARY_DIR}/rtneural_layer_bench"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rtneural_layer_bench> ${PROJECT_BINARY_DIR}/rtneural_layer_bench)

#add_executable(rtneural_model_bench model_bench.cpp)
#target_link_libraries(rtneural_model_bench LINK_PUBLIC RTNeural)
    
#add_custom_command(TARGET rtneural_model_bench
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:rtneural_model_bench> to ${PROJECT_BINARY_DIR}/rtneural_model_bench"
#    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rtneural_model_bench> ${PROJECT_BINARY_DIR}/rtneural_model_bench)
