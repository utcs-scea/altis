from datetime import datetime
from optparse import OptionParser
import numpy as np
import random
import sys

MIN_NODES = 20
MAX_NODES = 2**31
MIN_EDGES = 2
MAX_EDGES = 8
MIN_WEIGHT = 1
MAX_WEIGHT = 10

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-n', '--nodes', type="int", default=100000, help='Number of nodes in the graph')
    (options, args) = parser.parse_args()

    # check options
    if options.nodes < MIN_NODES:
        print("Error: Number of nodes must be greater than 20.")
        sys.exit()
    if options.nodes > MAX_NODES:
        print("Error: Number of nodes must be less than 2^31.")
        sys.exit()

    print("Generating graph with %d nodes" % options.nodes)

    # seed random number generator
    random.seed(datetime.now())
    edge_map = {}
    cost_map = {}

    # for each node, generate number of edges
    for i in range(options.nodes):
        edge_map[i] = []
        cost_map[i] = []
        num_edges = random.randint(MIN_EDGES, MAX_EDGES)
        # for each edge, generate destination and cost
        for j in range(num_edges):
            dest = np.random.uniform(0, options.nodes)
            cost  = random.randint(MIN_WEIGHT, MAX_WEIGHT)
            edge_map[i].append(dest)
            cost_map[i].append(cost)

    total_edges = 0
    # write graph to file
    with open("bfs_%d" % options.nodes, 'w') as f:
        # write total number of nodes
        f.write("%d\n" % options.nodes)
        # for each node, write number of edges
        for i in range(options.nodes):
            curr_edges = len(edge_map[i])
            f.write("%d %d\n" % (total_edges, curr_edges))
            total_edges += curr_edges
        source = np.random.uniform(0, options.nodes)
        # write source node
        f.write("%d\n" % source)
        # write total number of edges
        f.write("%d\n" % total_edges)
        # for each edge, write destination and cost
        for i in range(options.nodes):
            for j in range(len(edge_map[i])):
                f.write("%d %d\n" % (edge_map[i][j], cost_map[i][j]))
