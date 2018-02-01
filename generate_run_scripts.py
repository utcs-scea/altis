import os
import sys

level0 = ['BusSpeedDownload', 'BusSpeedReadback', 'DeviceMemory', 'MaxFlops']

# TODO: Get problem size based on device capabilities
def get_problem_size():
    return 1

# Write header
# f: file to write to
def write_header(f):
    f.write('#!/bin/bash\n')

# Write command to execute the benchmark
# f:     file to write to
# b_dir: directory of benchmark
# b:     name of benchmark
# sz:    problem size
def write_cmd(f, b_dir, b, sz):
    exec_name = os.path.join(b_dir, b)
    result_name = os.path.join(result_dir, b)
    f.write('.%s --outfile %s -s %d\n' % (exec_name, result_name, sz))

if __name__ == '__main__':
    # Get root directory
    if len(sys.argv) == 1:
        prefix = os.getcwd()
    else:
        prefix = sys.argv[1]
    
    # Path of results folder
    result_dir = os.path.join(prefix, 'results')
    
    # Problem size
    sz = get_problem_size()
    
    level_dir = os.path.join(prefix, 'src/cuda/level0')
    for b in level0:
        b_dir = os.path.join(level_dir, b)
        with open(os.path.join(b_dir, 'run'), 'w') as f:
            write_header(f)
            write_cmd(f, b_dir, b, sz)

