#!/usr/bin/python3

#
#  Parser.
#

import networkx as nx
import sys

# Reads graphs.
def read_graph (file):
    graph     = nx.Graph()
    terminals = nx.Graph()
    with open(file) as input:
        graph.add_weighted_edges_from([(int(e.split()[1]),
                                        int(e.split()[2]),
                                        int(e.split()[3]))
                                        for e in input if e.startswith("E ")])
    with open(file) as input:
        terminals.add_nodes_from([int(u.split()[1])
                                  for u in input if u.startswith("T ")])

    return [graph,terminals]


def read_graph_stdin ():
    graph     = nx.Graph()
    terminals = nx.Graph()

    for e in sys.stdin:
        if e.startswith("E "):
            graph.add_weighted_edges_from([(int(e.split()[1]),
                                            int(e.split()[2]),
                                            int(e.split()[3]))])
        if e.startswith("T "):
            terminals.add_nodes_from([int(e.split()[1])])

    return [graph,terminals]

if __name__ == '__main__':
    g = read_graph_stdin()

