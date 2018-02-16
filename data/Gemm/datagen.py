from optparse import OptionParser
import random
import sys

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s', '--seed', type="int", default=0, help='Seed for random number generator')
    parser.add_option('-k', '--kib', type="int", default=1, help='Size of matrix (in Kibibytes)')
    (options, args) = parser.parse_args()

    # check options
    if options.seed < 0:
        print("Error: Seed must be positive.")
        sys.exit()
    if options.kib <= 0:
        print("Error: Data size must be positive.")
        sys.exit()

    # if provided, seed random number generator
    if options.seed > 0:
        random.seed(options.seed)

    # fill method in gemm.cpp uses this value to generate elements
    maxi = 31

    with open('gemm_%d' % options.kib, 'w') as f:
        # write header line
        f.write('gemm_matrix %d\n' % options.kib)
        # number of floats
        n = options.kib * 1024 // 4
        for i in range(n*n):
            val_a = ((random.uniform(0, 100) % (maxi*2+1))-maxi)/(maxi+1.0)
            val_b = ((random.uniform(0, 100) % (maxi*2+1))-maxi)/(maxi+1.0)
            val_c = ((random.uniform(0, 100) % (maxi*2+1))-maxi)/(maxi+1.0)
            f.write('%0.4f %0.4f %0.4f\n' % (val_a, val_b, val_c))


