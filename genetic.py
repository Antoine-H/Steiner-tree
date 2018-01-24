
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



def first_solution(graph,terminals):
    graph_t = nx.Graph()
    too_add = []
    approx_spanning = nx.Graph()
    for i in range(len(terminals.nodes())):
        for j in range(len(terminals.nodes())-i-1):
            w = nx.shortest_path_length(graph,terminals.nodes()[i], terminals.nodes()[j],"weight")
            too_add.append((terminals.nodes()[i],terminals.nodes()[j],w))
    graph_t.add_weighted_edges_from(too_add)
    spanning_tree = nx.minimum_spanning_tree(graph_t)
    for (i,j) in spanning_tree.edges():
        path = nx.shortest_path(graph,i, j,"weight")
        for i in range(len(path)-1):
            approx_spanning.add_edge(path[i],path[i+1])
    return(approx_spanning)




if __name__ == '__main__':
    g = read_graph("Heuristic/instance001.gr")

