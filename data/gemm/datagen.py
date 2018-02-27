from optparse import OptionParser
import random
import sys

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-k', '--kib', type="int", default=1, help='Size of matrix (in Kibibytes)')
    (options, args) = parser.parse_args()

    # check options
    if options.kib <= 0:
        print("Error: Data size must be positive.")
        sys.exit()

    # fill method in gemm.cpp uses this value to generate elements
    maxi = 31

    with open('gemm_%d' % options.kib, 'w') as f:
        # write header line
        f.write('%d\n' % options.kib)
        # number of floats
        n = options.kib * 1024 // 4
        for i in range(n*n):
            val_a = ((random.randint(0, sys.maxsize) % (maxi*2+1))-maxi)/(maxi+1.0)
            val_b = ((random.randint(0, sys.maxsize) % (maxi*2+1))-maxi)/(maxi+1.0)
            val_c = ((random.randint(0, sys.maxsize) % (maxi*2+1))-maxi)/(maxi+1.0)
            f.write('%0.4f %0.4f %0.4f\n' % (val_a, val_b, val_c))


