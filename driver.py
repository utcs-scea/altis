import os
import subprocess
import sys
from optparse import OptionParser

suite = ['BusSpeedDownload', 'BusSpeedReadback', 'DeviceMemory', 'MaxFlops', 'Sort', 'Spmv', 'Stencil2D', 'Gemm', 'Backprop', 'lavaMD']
suite_map = {'BusSpeedDownload':0, 'BusSpeedReadback':0, 'DeviceMemory':0, 'MaxFlops':0, 'Sort':1, 'Spmv':1, 'Stencil2D':1, 'Gemm':1, 'Backprop':2, 'lavaMD': 2}

# TODO: Get problem size based on device capabilities
def get_problem_size():
    return 1

def run_benchmark(options, b):
    print('*****')
    print('Running: %s...' % b)
    # Path of results file
    result_path = os.path.join(options.prefix, 'results/%s' % (b))
    # Path of executble
    exec_path = os.path.join(options.prefix, 'src/cuda/level%d/%s/%s' % (suite_map[b], b, b))
    # Execute benchmark with options
    p = subprocess.Popen([exec_path, '-s', str(options.size), '-o', result_path, '-d', str(options.device)])
    (stdoutdata, stderrdata) = p.communicate()
    print('Done.')

if __name__ == '__main__':
    # Options
    parser = OptionParser()
    parser.add_option('-p', '--prefix', default=os.getcwd(), help='location of Mirovia root, defaults to current working directory')
    parser.add_option('-d', '--device', default=0, help='device to run the benchmarks on')
    parser.add_option('-s', '--size', default=0, help='problem size')
    parser.add_option('-b', '--benchmark', default='all', help='comma-separated list of benchmarks to run, or \'all\' to run entire suite, defaults to \'all\'')
    
    # Parse options
    (options, args) = parser.parse_args()

    # Problem size
    if int(options.size) == 0:
        options.size = get_problem_size()
    if options.size > 4 or options.size < 1:
        print('Error: Problem size must be between 1-4')
        sys.exit(1)
    
    # Benchmarks
    if options.benchmark == 'all':
        benchmarks = suite
    else:
        benchmarks = options.benchmark.split(',')

    # Run benchmarks
    print('Running Mirovia driver...')
    print('Prefix: %s' % (options.prefix))
    print('Device: %s' % (options.device))
    print('Problem size: %s' % (options.size))
    for b in benchmarks:
        if b not in suite_map:
            print('Error: Benchmark %s does not exist')
            sys.exit(1)
        run_benchmark(options, b)

