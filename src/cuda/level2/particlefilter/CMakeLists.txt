cmake_minimum_required(VERSION 3.8)

add_library(particlefilterLib ex_particle_CUDA_naive_seq.cu)
target_link_libraries(particlefilterLib -lm)

add_library(particlefilterfloatLib ex_particle_CUDA_float_seq.cu)
target_link_libraries(particlefilterfloatLib -lm)
