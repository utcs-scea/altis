from datetime import datetime
from optparse import OptionParser
import random
import sys

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s', '--size', type="int", default=100000, help='Number of elements')
    (options, args) = parser.parse_args()

    # check options
    if options.size <= 0:
        print("Error: Number of elements must be positive.")
        sys.exit()

    print("Generating input with %d elements" % options.size)

    random.seed(datetime.now())
    with open('cfd_%d' % options.size, 'w') as f:
        # write header line
        f.write('%d\n' % options.size)
        # number of floats
        for i in range(options.size):
            f.write('%0.7f   ' % random.uniform(0, 1))
            for j in range(4):
                f.write('%d ' % random.uniform(i - 10, i + 10))
                for k in range(3):
                    f.write('%0.7f ' % random.uniform(-0.5, 0.5))
            f.write('\n')

