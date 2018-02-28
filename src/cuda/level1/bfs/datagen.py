from optparse import OptionParser
import random
import sys

MIN_NODES = 20
MAX_NODES = sys.maxsize
MIN_EDGES = 2
MAX_INIT_EDGES = 4
MIN_WEIGHT = 1
MAX_WEIGHT = 10

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-n', '--nodes', type="int", default=1, help='Number of nodes in the graph')
    (options, args) = parser.parse_args()

    # check options
    if options.nodes <= 0:
        print("Error: Number of nodes must be positive.")
        sys.exit()

    print(MAX_NODES)
