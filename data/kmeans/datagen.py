from datetime import datetime
from optparse import OptionParser
import numpy as np
import random
import sys

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-n', '--numObjects', type="int", default=10000, help='Number of objects')
    parser.add_option('-a', '--numAttributes', type="int", default=30, help='Number of attributes of each object')
    parser.add_option('-f', '--float', action="store_true", default=False, help="Features are floats (integers by default)")
    (options, args) = parser.parse_args()

    # check options
    if options.numObjects <= 0:
        print("Error: Number of objects must be positive.")
        sys.exit()
    if options.numAttributes <= 0:
        print("Error: Number of attributes must be positive.")
        sys.exit()

    n = options.numObjects
    a = options.numAttributes
    f = options.float
    print("Generating %d objects with %d %s attributes each" % (n,a, 'float' if f else 'integer'))

    # seed random number generator
    random.seed(datetime.now())

    if(f):
        name = 'kmeans_f_%d_%d' % (n,a)
    else:
        name = 'kmeans_%d_%d' % (n,a)

    with open(name, 'w') as fn:
        # write number of objects and number of features
        fn.write('%d %d\n' % (n,a))
        for i in range(n):
            # write ID of object (ignored in benchmark)
            fn.write('%d' % i)
            for j in range(a):
                # write value of attribute
               if(f):
                   fn.write(' %0.4f' % random.random())
               else:
                   fn.write(' %d' % random.randint(0, 255))
            fn.write('\n')

