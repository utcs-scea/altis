from optparse import OptionParser
import random

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-c', '--cols', type="int", default=100, help='How many columns in the matrix')
    parser.add_option('-r', '--rows', type="int", default=100, help='How many rows in the matrix')
    parser.add_option('-m', '--maxval', type="int", default=100, help='Maximum non-zero value in matrix')
    parser.add_option('-p', '--pattern', action='store_true', help='Flag allowing benchmark to generate matrix values')
    (options, args) = parser.parse_args()

    n = options.rows * options.cols // 10
    with open('spmv_%dx%d' % (options.rows, options.cols), 'w') as f:
        field = 'pattern' if options.pattern else 'no_pattern'
        # write header line
        f.write('%d %s %s %s\n' % (n, 'matrix', 'coordinate', field))
        # write dimensions of matrix, number of non-zero elements
        f.write('%d %d %d\n' % (options.rows, options.cols, n))
        points = set()
        while len(points) < n:
            row = random.randint(0, options.rows-1)
            col = random.randint(0, options.cols-1)
            if (row, col) in points:
                continue
            points.add((row, col))
            if options.pattern:
                f.write('%d %d\n' % (row, col))
            else:
                val = random.uniform(0, options.maxval)
                f.write('%d %d %0.2f\n' % (row, col, val))

