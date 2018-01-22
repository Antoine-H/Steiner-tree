
#
#  Parser.
#

import networkx as nx

# Reads graphs.
def read_graph (file):
    graph     = nx.Graph()
    terminals = nx.Graph()
    with open(file) as input:
        graph.add_weighted_edges_from([(int(e.split()[1]),
                                        int(e.split()[2]),
                                        int(e.split()[3]))
                                        for e in input if e.startswith("E ")])
        terminals.add_nodes_from([int(u.split()[1])
                                  for u in input if u.startswith("T ")])
    return [graph,terminals]

# Main

g = read_graph("Heuristic/instance001.gr")

