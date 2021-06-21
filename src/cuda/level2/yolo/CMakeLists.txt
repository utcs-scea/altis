cmake_minimum_required(VERSION 3.8)

add_compile_definitions(GPU)
set(darknetSourceDir ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(darketSources 
    ${darknetSourceDir}/activation_kernels.cu
    ${darknetSourceDir}/activation_layer.c
    ${darknetSourceDir}/activations.c
    ${darknetSourceDir}/avgpool_layer.c
    ${darknetSourceDir}/avgpool_layer_kernels.cu
    ${darknetSourceDir}/batchnorm_layer.c
    ${darknetSourceDir}/blas.c
    ${darknetSourceDir}/blas_kernels.cu
    ${darknetSourceDir}/box.c
    ${darknetSourceDir}/col2im.c
    ${darknetSourceDir}/col2im_kernels.cu
    ${darknetSourceDir}/compare.c
    ${darknetSourceDir}/connected_layer.c
    ${darknetSourceDir}/convolutional_kernels.cu
    ${darknetSourceDir}/convolutional_layer.c
    ${darknetSourceDir}/cost_layer.c
    ${darknetSourceDir}/crnn_layer.c
    ${darknetSourceDir}/crop_layer.c
    ${darknetSourceDir}/crop_layer_kernels.cu
    ${darknetSourceDir}/cuda.c
    ${darknetSourceDir}/data.c
    ${darknetSourceDir}/deconvolutional_kernels.cu
    ${darknetSourceDir}/deconvolutional_layer.c
    ${darknetSourceDir}/demo.c
    ${darknetSourceDir}/detection_layer.c
    ${darknetSourceDir}/dropout_layer.c
    ${darknetSourceDir}/dropout_layer_kernels.cu
    ${darknetSourceDir}/gemm.c
    ${darknetSourceDir}/gru_layer.c
    ${darknetSourceDir}/im2col.c
    ${darknetSourceDir}/im2col_kernels.cu
    ${darknetSourceDir}/image.c
    ${darknetSourceDir}/iseg_layer.c
    ${darknetSourceDir}/l2norm_layer.c
    ${darknetSourceDir}/layer.c
    ${darknetSourceDir}/list.c
    ${darknetSourceDir}/local_layer.c
    ${darknetSourceDir}/logistic_layer.c
    ${darknetSourceDir}/lstm_layer.c
    ${darknetSourceDir}/matrix.c
    ${darknetSourceDir}/maxpool_layer.c
    ${darknetSourceDir}/maxpool_layer_kernels.cu
    ${darknetSourceDir}/network.c
    ${darknetSourceDir}/normalization_layer.c
    ${darknetSourceDir}/option_list.c
    ${darknetSourceDir}/parser.c
    ${darknetSourceDir}/region_layer.c
    ${darknetSourceDir}/reorg_layer.c
    ${darknetSourceDir}/rnn_layer.c
    ${darknetSourceDir}/route_layer.c
    ${darknetSourceDir}/shortcut_layer.c
    ${darknetSourceDir}/softmax_layer.c
    ${darknetSourceDir}/tree.c
    ${darknetSourceDir}/upsample_layer.c
    ${darknetSourceDir}/utils.c
    ${darknetSourceDir}/yolo_layer.c
)

add_library(darknetLib ${darketSources})
target_include_directories(darknetLib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(darknetLib -lcuda -lm -pthread -lcudart -lcublas -lcublas -lcurand)
if(${CMAKE_VERSION} VERSION_LESS "3.18.0") 
    target_compile_options(darknetLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${ARCH_FLAGS} -rdc=true>)
else()
    target_compile_options(darknetLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -rdc=true>)
endif()
set_target_properties(sradLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

