cmake_minimum_required(VERSION 3.8)

add_compile_definitions(GPU)
set(darknetSourceDir ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(darketSources
    ${darknetSourceDir}/activation_layer.c
    ${darknetSourceDir}/activations.c
    ${darknetSourceDir}/art.c
    ${darknetSourceDir}/avgpool_layer.c
    ${darknetSourceDir}/batchnorm_layer.c
    ${darknetSourceDir}/blas.c
    ${darknetSourceDir}/box.c
    ${darknetSourceDir}/captcha.c
    ${darknetSourceDir}/cifar.c
    ${darknetSourceDir}/classifier.c
    ${darknetSourceDir}/coco.c
    ${darknetSourceDir}/col2im.c
    ${darknetSourceDir}/compare.c
    ${darknetSourceDir}/connected_layer.c
    ${darknetSourceDir}/conv_lstm_layer.c
    ${darknetSourceDir}/convolutional_layer.c
    ${darknetSourceDir}/cost_layer.c
    ${darknetSourceDir}/cpu_gemm.c
    ${darknetSourceDir}/crnn_layer.c
    ${darknetSourceDir}/crop_layer.c
    ${darknetSourceDir}/dark_cuda.c
    ${darknetSourceDir}/data.c
    ${darknetSourceDir}/deconvolutional_layer.c
    ${darknetSourceDir}/demo.c
    ${darknetSourceDir}/detection_layer.c
    ${darknetSourceDir}/detector.c
    ${darknetSourceDir}/dice.c
    ${darknetSourceDir}/dropout_layer.c
    ${darknetSourceDir}/gaussian_yolo_layer.c
    ${darknetSourceDir}/gemm.c
    ${darknetSourceDir}/getopt.c
    ${darknetSourceDir}/gettimeofday.c
    ${darknetSourceDir}/go.c
    ${darknetSourceDir}/gru_layer.c
    ${darknetSourceDir}/im2col.c
    ${darknetSourceDir}/image.c
    ${darknetSourceDir}/layer.c
    ${darknetSourceDir}/list.c
    ${darknetSourceDir}/local_layer.c
    ${darknetSourceDir}/lstm_layer.c
    ${darknetSourceDir}/matrix.c
    ${darknetSourceDir}/maxpool_layer.c
    ${darknetSourceDir}/network.c
    ${darknetSourceDir}/nightmare.c
    ${darknetSourceDir}/normalization_layer.c
    ${darknetSourceDir}/option_list.c
    ${darknetSourceDir}/parser.c
    ${darknetSourceDir}/region_layer.c
    ${darknetSourceDir}/reorg_layer.c
    ${darknetSourceDir}/reorg_old_layer.c
    ${darknetSourceDir}/representation_layer.c
    ${darknetSourceDir}/rnn.c
    ${darknetSourceDir}/rnn_layer.c
    ${darknetSourceDir}/rnn_vid.c
    ${darknetSourceDir}/route_layer.c
    ${darknetSourceDir}/sam_layer.c
    ${darknetSourceDir}/scale_channels_layer.c
    ${darknetSourceDir}/shortcut_layer.c
    ${darknetSourceDir}/softmax_layer.c
    ${darknetSourceDir}/super.c
    ${darknetSourceDir}/swag.c
    ${darknetSourceDir}/tag.c
    ${darknetSourceDir}/tree.c
    ${darknetSourceDir}/upsample_layer.c
    ${darknetSourceDir}/utils.c
    ${darknetSourceDir}/voxel.c
    ${darknetSourceDir}/writing.c
    ${darknetSourceDir}/yolo.c
    ${darknetSourceDir}/yolo_layer.c
)
set_source_files_properties(${darketSources} PROPERTIES LANGUAGE CUDA)

set(darketCudaSources
    ${darknetSourceDir}/activation_kernels.cu
    ${darknetSourceDir}/maxpool_layer_kernels.cu
    ${darknetSourceDir}/network_kernels.cu
    ${darknetSourceDir}/im2col_kernels.cu
    ${darknetSourceDir}/dropout_layer_kernels.cu
    ${darknetSourceDir}/deconvolutional_kernels.cu
    ${darknetSourceDir}/crop_layer_kernels.cu
    ${darknetSourceDir}/convolutional_kernels.cu
    ${darknetSourceDir}/blas_kernels.cu
    ${darknetSourceDir}/avgpool_layer_kernels.cu
    ${darknetSourceDir}/col2im_kernels.cu
)

set(darketCXXSources
    ${darknetSourceDir}/image_opencv.cpp
    ${darknetSourceDir}/yolo_v2_class.cpp
    # ${darknetSourceDir}/yolo_console_dll.cpp
    ${darknetSourceDir}/http_stream.cpp
)
set_source_files_properties(${darketCXXSources} PROPERTIES LANGUAGE CUDA)

add_library(darknetLib ${darketSources} ${darketCudaSources} ${darketCXXSources})
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
set_target_properties(darknetLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
