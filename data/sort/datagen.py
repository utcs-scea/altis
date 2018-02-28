from datetime import datetime
from optparse import OptionParser
import random
import sys

MIN_VAL = 0
MAX_VAL = 1024

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s', '--size', type="int", default=1024, help='Number of elements in the array')
    (options, args) = parser.parse_args()

    # check options
    if options.size <= 0:
        print("Error: Number of elements must be positive.")
        sys.exit()

    print("Generating array with %d elements" % options.size)

    random.seed(datetime.now())
    with open('sort_%d' % options.size, 'w') as f:
        # write header line
        f.write('%d\n' % options.size)
        # number of floats
        for i in range(options.size):
            f.write('%0.4f\n' % random.uniform(MIN_VAL, MAX_VAL))

