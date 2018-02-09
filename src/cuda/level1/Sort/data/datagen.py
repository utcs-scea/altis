from optparse import OptionParser
import random

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s', '--size', default = 16, help = 'How many millions of elements to sort')
    (options, args) = parser.parse_args()
    size = int(options.size) * 1024 * 1024

    with open("sort_" + str(options.size), "w") as f:
        f.write(str(size) + '\n')
        for _ in range(size):
            f.write(str(random.randint(1, 1024)) + '\n')
