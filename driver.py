import os
import subprocess
import sys
from optparse import OptionParser

suite = ['busspeeddownload', 'busspeedreadback', 'devicememory', 'maxflops', 'bfs', 'gemm', 'pathfinder', 'sort', 'cfd', 'dwt2d', 'fdtd2d', 'gups', 'kmeans', 'lavamd', 'mandelbrot', 'nw', 'particlefilter', 'srad', 'where']
suite_map = {'busspeeddownload':0, 'busspeedreadback':0, 'devicememory':0, 'maxflops':0, 'bfs':1, 'gemm':1, 'pathfinder':1, 'sort':1, 'cfd':2, 'dwt2d':2, 'fdtd2d':2, 'gups':2, 'kmeans':2, 'lavamd':2, 'mandelbrot':2, 'nw':2, 'particlefilter':2, 'srad':2, 'where':2}

def run_benchmark(options, b):
    print('*****')
    # Path of results file
    result_path = os.path.join(options.prefix, 'results/%s' % (b))
    # Path of executble
    paths = []
    if b == 'particlefilter':
        paths.append(os.path.join(options.prefix, 'src/cuda/level%d/%s/%s_float' % (suite_map[b], b, b)))
        paths.append(os.path.join(options.prefix, 'src/cuda/level%d/%s/%s_naive' % (suite_map[b], b, b)))
    if b == 'cfd':
        paths.append(os.path.join(options.prefix, 'src/cuda/level%d/%s/%s' % (suite_map[b], b, b)))
        paths.append(os.path.join(options.prefix, 'src/cuda/level%d/%s/%s_double' % (suite_map[b], b, b)))
    else:
        paths.append(os.path.join(options.prefix, 'src/cuda/level%d/%s/%s' % (suite_map[b], b, b)))
    # Execute benchmark with options
    for path in paths:
        print(path)
        if options.verbose:
            p = subprocess.Popen([path, '-s', str(options.size), '-o', result_path, '-d', str(options.device)])
        else:
            p = subprocess.Popen([path, '-s', str(options.size), '-o', result_path, '-d', str(options.device), '-q'])
        (stdoutdata, stderrdata) = p.communicate()

if __name__ == '__main__':
    # Options
    parser = OptionParser()
    parser.add_option('-p', '--prefix', default=os.getcwd(), help='location of Altis root, defaults to current working directory')
    parser.add_option('-d', '--device', default=0, help='device to run the benchmarks on')
    parser.add_option('-s', '--size', default=0, help='problem size')
    parser.add_option('-b', '--benchmark', default='all', help='comma-separated list of benchmarks to run, or \'all\' to run entire suite, defaults to \'all\'')
    parser.add_option('-v', '--verbose', action='store_true', help='enable verbose output')
    parser.add_option('\0', '--shifts', default=20, help='size of update table')   # Only for GUPS
    
    # Parse options
    (options, args) = parser.parse_args()

    # Problem size
    if int(options.size) == 0:
        options.size = 1
    if options.size > 4 or options.size < 1:
        print('Error: Problem size must be between 1-4')
        sys.exit(1)
    
    # Benchmarks
    if options.benchmark == 'all':
        benchmarks = suite
    else:
        benchmarks = options.benchmark.split(',')

    # Run benchmarks
    print('Running Altis driver...')
    print('Prefix: %s' % (options.prefix))
    print('Device: %s' % (options.device))
    print('Problem size: %s' % (options.size))
    for b in benchmarks:
        if b not in suite_map:
            print('Error: Benchmark %s does not exist')
            sys.exit(1)
        run_benchmark(options, b)

