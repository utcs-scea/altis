#!/bin/bash

#for f in devicememory maxflops
#do
#for i in 1 2 3 4
#do
#nvprof --metrics cf_fu_utilization,tex_fu_utilization,ldst_fu_utilization,double_precision_fu_utilization,special_fu_utilization,single_precision_fu_utilization,flop_count_dp,flop_count_sp,dram_utilization,tex_utilization,shared_utilization,inst_fp_32,inst_fp_64,inst_integer,inst_bit_convert,inst_control,inst_compute_ld_st,inst_misc,inst_inter_thread_communication,l2_utilization,sysmem_utilization --log-file analysis/zemaitis/$f/$i ./src/cuda/level0/$f/$f -s $i -n 1
#done
#done

#for f in gemm pathfinder sort bfs
for f in sort pathfinder
do
for i in 1 2 3 4
do
nvprof --metrics cf_fu_utilization,tex_fu_utilization,ldst_fu_utilization,double_precision_fu_utilization,special_fu_utilization,single_precision_fu_utilization,flop_count_dp,flop_count_sp,dram_utilization,tex_utilization,shared_utilization,inst_fp_32,inst_fp_64,inst_integer,inst_bit_convert,inst_control,inst_compute_ld_st,inst_misc,inst_inter_thread_communication,l2_utilization,sysmem_utilization --log-file analysis/zemaitis/$f/$i ./src/cuda/level1/$f/$f -s $i -n 1
done
done

#for f in cfd kmeans lavamd nw srad
#do
#for i in 1 2 3 4
#do
#nvprof --metrics cf_fu_utilization,tex_fu_utilization,ldst_fu_utilization,double_precision_fu_utilization,special_fu_utilization,single_precision_fu_utilization,flop_count_dp,flop_count_sp,dram_utilization,tex_utilization,shared_utilization,inst_fp_32,inst_fp_64,inst_integer,inst_bit_convert,inst_control,inst_compute_ld_st,inst_misc,inst_inter_thread_communication,l2_utilization,sysmem_utilization --log-file analysis/$f/$i ./src/cuda/level2/$f/$f -s $i -n 1
#done
#done

#cf_fu_utilization:  The utilization level of the multiprocessor function units that execute control-flow instructions on a scale of 0 to 10
#tex_fu_utilization:  The utilization level of the multiprocessor function units that execute global, local and texture memory instructions on a scale of 0 to 10
#ldst_fu_utilization:  The utilization level of the multiprocessor function units that execute shared load, shared store and constant load instructions
#double_precision_fu_utilization:  The utilization level of the multiprocessor function units that execute double-precision floating-point instructions
#special_fu_utilization:  The utilization level of the multiprocessor function units that execute sin, cos, ex2, popc, flo, and similar instructions on a scale of 0 to 10
#single_precision_fu_utilization:  The utilization level of the multiprocessor function units that execute single-precision floating-point instructions and integer instructions
#flop_count_dp:  Number of double-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count.
#flop_count_sp:  Number of single-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count. The count does not include special operations.
#dram_utilization:  The utilization level of the device memory relative to the peak utilization on a scale of 0 to 10
#tex_utilization:  The utilization level of the unified cache relative to the peak utilization
#shared_utilization:  The utilization level of the shared memory relative to peak utilization
#inst_fp_32:  Number of single-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)
#inst_fp_64:  Number of double-precision floating-point instructions executed by non-predicated threads (arithmetic, compare, etc.)
#inst_integer:  Number of integer instructions executed by non-predicated threads
#inst_bit_convert:  Number of bit-conversion instructions executed by non-predicated threads
#inst_control:  Number of control-flow instructions executed by non-predicated threads (jump, branch, etc.)
#inst_compute_ld_st:  Number of compute load/store instructions executed by non-predicated threads
#inst_misc:  Number of miscellaneous instructions executed by non-predicated threads
#inst_inter_thread_communication:  Number of inter-thread communication instructions executed by non-predicated threads
#l2_utilization:  The utilization level of the L2 cache relative to the peak utilization on a scale of 0 to 10
#sysmem_utilization:  The utilization level of the system memory relative to the peak utilization
#sysmem_read_utilization:  The read utilization level of the system memory relative to the peak utilization on a scale of 0 to 10
#sysmem_write_utilization:  The write utilization level of the system memory relative to the peak utilization on a scale of 0 to 10
