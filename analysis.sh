#!/bin/bash

#for f in devicememory maxflops
#do
#echo $f
#nvprof --metrics flop_count_dp,flop_count_sp,inst_fp_32,inst_fp_64,inst_integer,inst_bit_convert,inst_control,inst_compute_ld_st,inst_misc,inst_inter_thread_communication,sm_efficiency,achieved_occupancy,ipc,branch_efficiency,warp_execution_efficiency,shared_store_transactions,shared_load_transactions,local_load_transactions,local_store_transactions,gld_transactions,gst_transactions,dram_read_transactions,dram_write_transactions,flop_count_sp_special,inst_executed,cf_executed,ldst_executed --log-file analysis/zemaitis/$f/1 ./src/cuda/level0/$f/$f -n 1 -d 1
#nvprof --metrics sm_efficiency,achieved_occupancy,ipc,branch_efficiency,warp_execution_efficiency,shared_store_transactions,shared_load_transactions,local_load_transactions,local_store_transactions,gld_transactions,gst_transactions,dram_read_transactions,dram_write_transactions,flop_count_sp_special,inst_executed,cf_executed,ldst_executed --log-file analysis/zerberus/$f/1 ./src/cuda/level0/$f/$f -n 1
#done

for f in pathfinder sort bfs gemm
do
    echo $f
sudo /usr/local/cuda-10.0/bin/nvprof --metrics dram_utilization,l2_utilization,shared_utilization,tex_utilization,cf_fu_utilization,double_fu_utilization,single_fu_utilization,ldst_fu_utilization,special_fu_utilization,tex_fu_utilization --log-file analysis/zemaitis/$f/4 ./src/cuda/level1/$f/$f -s 4 -n 1 -d 1
#sudo /usr/local/cuda-10.0/bin/nvprof --metrics dram_utiliztion, l2_utilization, shared_utilization, tex_utilization, cf_fu_utilization, double_fu_utilization, single_fu_utilization, ldst_fu_utilization, special_fu_utilization, tex_fu_utilization ./src/cuda/level1/$f/$f -s 4 -n 1 -d 1

#nvprof --metrics sm_efficiency,achieved_occupancy,ipc,branch_efficiency,warp_execution_efficiency,shared_store_transactions,shared_load_transactions,local_load_transactions,local_store_transactions,gld_transactions,gst_transactions,dram_read_transactions,dram_write_transactions,flop_count_sp_special,inst_executed,cf_executed,ldst_executed --log-file analysis/zerberus/$f/$i ./src/cuda/level1/$f/$f -s $i -n 1
done

for f in cfd dwt2d kmeans lavamd mandelbrot nw srad where
do
#nvprof --metrics flop_count_dp,flop_count_sp,inst_fp_32,inst_fp_64,inst_integer,inst_bit_convert,inst_control,inst_compute_ld_st,inst_misc,inst_inter_thread_communication,sm_efficiency,achieved_occupancy,ipc,branch_efficiency,warp_execution_efficiency,shared_store_transactions,shared_load_transactions,local_load_transactions,local_store_transactions,gld_transactions,gst_transactions,dram_read_transactions,dram_write_transactions,flop_count_sp_special,inst_executed,cf_executed,ldst_executed --log-file analysis/zemaitis/$f/4 ./src/cuda/level2/$f/$f -s 4 -n 1 -d 1
sudo /usr/local/cuda-10.0/bin/nvprof --metrics  dram_utiliztion,l2_utilization,shared_utilization,tex_utilization,cf_fu_utilization,double_fu_utilization,single_fu_utilization,ldst_fu_utilization,special_fu_utilization,tex_fu_utilization --log-file analysis/zemaitis/$f/4 ./src/cuda/level2/$f/$f -s 4 -n 1 -d 1
#nvprof --metrics sm_efficiency,achieved_occupancy,ipc,branch_efficiency,warp_execution_efficiency,shared_store_transactions,shared_load_transactions,local_load_transactions,local_store_transactions,gld_transactions,gst_transactions,dram_read_transactions,dram_write_transactions,flop_count_sp_special,inst_executed,cf_executed,ldst_executed --log-file analysis/zerberus/$f/$i ./src/cuda/level2/$f/$f -s $i -n 1
done

for f in naive float
do
#nvprof --metrics flop_count_dp,flop_count_sp,inst_fp_32,inst_fp_64,inst_integer,inst_bit_convert,inst_control,inst_compute_ld_st,inst_misc,inst_inter_thread_communication,sm_efficiency,achieved_occupancy,ipc,branch_efficiency,warp_execution_efficiency,shared_store_transactions,shared_load_transactions,local_load_transactions,local_store_transactions,gld_transactions,gst_transactions,dram_read_transactions,dram_write_transactions,flop_count_sp_special,inst_executed,cf_executed,ldst_executed --log-file analysis/zemaitis/particlefilter/$f/4 ./src/cuda/level2/particlefilter/particlefilter_$f -s 4 -n 1 -d 1
sudo /usr/local/cuda-10.0/bin/nvprof --metrics dram_utiliztion,l2_utilization,shared_utilization,tex_utilization,cf_fu_utilization,double_fu_utilization,single_fu_utilization,ldst_fu_utilization,special_fu_utilization,tex_fu_utilization --log-file analysis/zemaitis/particlefilter/$f/4 ./src/cuda/level2/particlefilter/particlefilter_$f -s 4 -n 1 -d 1
#nvprof --metrics sm_efficiency,achieved_occupancy,ipc,branch_efficiency,warp_execution_efficiency,shared_store_transactions,shared_load_transactions,local_load_transactions,local_store_transactions,gld_transactions,gst_transactions,dram_read_transactions,dram_write_transactions,flop_count_sp_special,inst_executed,cf_executed,ldst_executed --log-file analysis/zerberus/particlefilter/$f/$i ./src/cuda/level2/particlefilter/particlefilter_$f -s $i -n 1
done


#sm_efficiency:  The percentage of time at least one warp is active on a specific multiprocessor
#achieved_occupancy:  Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor
#ipc:  Instructions executed per cycle
#branch_efficiency:  Ratio of non-divergent branches to total branches
#warp_execution_efficiency:  Ratio of the average active threads per warp to the maximum number of threads per warp supported on a multiprocessor
#shared_store_transactions:  Number of shared memory store transactions
#shared_load_transactions:  Number of shared memory load transactions
#local_load_transactions:  Number of local memory load transactions
#local_store_transactions:  Number of local memory store transactions
#gld_transactions:  Number of global memory load transactions
#gst_transactions:  Number of global memory store transactions
#dram_read_transactions:  Device memory read transactions
#dram_write_transactions:  Device memory write transactions
#flop_count_sp_special:  Number of single-precision floating-point special operations executed by non-predicated threads.
#inst_executed:  The number of instructions executed
#cf_executed:  Number of executed control-flow instructions
#ldst_executed:  Number of executed local, global, shared and texture memory load and store instructions


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
