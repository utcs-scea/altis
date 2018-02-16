from optparse import OptionParser
import random

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s', '--seed', type="int", default=0, help='Seed for random number generator')
    parser.add_option('-s', '--size', default = 16, help = 'How many millions of elements to sort')
    (options, args) = parser.parse_args()

    # check options
    if options.seed < 0:
        print("Error: Seed must be positive.")
        sys.exit()
    if options.size <= 0:
        print("Error: Array size must be positive.")
        sys.exit()

    # if provided, seed random number generator
    if options.seed > 0:
        random.seed(options.seed)

    size = int(options.size) * 1024 * 1024
    with open("sort_" + str(options.size), "w") as f:
        f.write(str(size) + '\n')
        for _ in range(size):
            f.write(str(random.randint(1, 1024)) + '\n')
