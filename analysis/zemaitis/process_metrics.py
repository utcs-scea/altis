import pprint
import pandas as pd

#benchmarks = ['devicememory', 'maxflops', 'bfs', 'gemm', 'sort', 'pathfinder', 'cfd', 'dwt2d', 'kmeans', 'lavamd', 'mandelbrot', 'nw', 'particlefilter', 'srad']
benchmarks = ['bfs', 'sort', 'pathfinder']
metrics = ['cf_fu_utilization','tex_fu_utilization','ldst_fu_utilization','double_precision_fu_utilization','special_fu_utilization','half_precision_fu_utilization','single_precision_fu_utilization','flop_count_dp','flop_count_sp','dram_utilization','tex_utilization','shared_utilization','inst_fp_32','inst_fp_64','inst_integer','inst_bit_convert','inst_control','inst_compute_ld_st','inst_misc','inst_inter_thread_communication','l2_utilization','sysmem_utilization']

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
    s = val.find('(')
    e = val.find(')')
    return val[s+1:e]

res = pd.DataFrame()
res['metric'] = metrics
res.set_index('metric', inplace=True)

for benchmark in benchmarks:
    for size in range(1,5):
        name = '%s_%s' % (benchmark, size)
        print(name)
        res[name] = 'n/a'
        # Open file
        try:
            f = open('%s/%d' % (benchmark, size))
        except:
            continue
        # Read intro lines
        [f.readline() for i in range(0,7)]
        # Start parsing
        kernel = ''
        for line in f:
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
