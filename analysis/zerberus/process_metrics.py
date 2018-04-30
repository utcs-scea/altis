import pprint
import pandas as pd

benchmarks = ['devicememory', 'maxflops', 'bfs', 'gemm', 'sort', 'pathfinder', 'cfd', 'dwt2d', 'kmeans', 'lavamd', 'mandelbrot', 'nw', 'particlefilter_float', 'particlefilter_naive', 'srad', 'where']
metrics = ['flop_count_dp','flop_count_sp','inst_fp_32','inst_fp_64','inst_integer','inst_bit_convert','inst_control','inst_compute_ld_st','inst_misc','inst_inter_thread_communication']
metrics_extra = ['sm_efficiency','achieved_occupancy','ipc,branch_efficiency','warp_execution_efficiency','shared_store_transactions','shared_load_transactions','local_load_transactions','local_store_transactions','gld_transactions','gst_transactions','dram_read_transactions','dram_write_transactions','flop_count_sp_special','inst_executed','cf_executed','ldst_executed']

kernel_delim = 'Kernel: '
col_1 = (0,0+len('Invocations'))
col_2 = (col_1[1], col_1[1]+len('                               Metric Name'))
col_3 = (col_2[1], col_2[1]+len('                            Metric Description'))
col_4 = (col_3[1], col_3[1]+len('         Min'))
col_5 = (col_4[1], col_4[1]+len('         Max'))
col_6 = (col_5[1], col_5[1]+len('         Avg'))

def is_kernel(line):
    return kernel_delim in line

def parse_kernel(line):
    return line.split(kernel_delim)[1].split('(')[0].strip()

def parse_val(val):
    return float(val[:-1])

res = pd.DataFrame()
res['metric'] = metrics + metrics_extra
res.set_index('metric', inplace=True)

for benchmark in benchmarks:
    for size in range(1,5):
        name = '%s_%s' % (benchmark, size)
        print(name)
        res[name] = 'n/a'
        # Open file
        try:
            if 'particlefilter' in name:
                f = open('particlefilter/%s/%d' % (benchmark.split('_')[1], size))
                f_e = open('extra/particlefilter/%s/%d' % (benchmark.split('_')[1], size))
            else:
                f = open('%s/%d' % (benchmark,size))
                f_e = open('extra/%s/%d' % (benchmark,size))
        except:
            print('cant open %s %d' % (benchmark,size))
            continue
        # Read intro lines
        [f.readline() for i in range(0,7)]
        [f_e.readline() for i in range(0,7)]
        # Start parsing
        lines = [line for line in f] + [line for line in f_e]
        kernel = ''
        for line in lines:
            # Parse kernel
            if is_kernel(line):
                kernel = parse_kernel(line)
                continue
            if not any([metric in line for metric in metrics]):
                print('\t' + line)
                continue
            # Parse metric
            metric = line[col_2[0]:col_2[1]].strip()
            val = line[col_6[0]:col_6[1]].strip()
            if not val.isdigit():
                val = parse_val(val)
            #print('\t' + name)
            if res.at[metric, name] == 'n/a':
                res.at[metric, name] = val
            elif res.at[metric, name] < val:
                res.at[metric, name] = val

res.to_csv(open('analysis.csv', 'w'))

print('Done.')
