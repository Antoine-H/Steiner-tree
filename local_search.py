
import networkx as nx

# Reads graphs.


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


def modification(graph, terminals):
	g2 = graph.copy()
	edges = shuffle(g2.edges(data=True)) #pour avoir un edge aleatoire
	for e in edges:
		for e in edges:
			g2.remove_edge(e)
			if(not(nx.is_connected(g2))):
				g2.add_adge(e)
	return(g2)


# Objective function
def gain (steiner):
    # Assuming that max eccentricity nodes are terminals
    w = nx.diameter(steiner)
    d = 0
    edges = g[0].edges(data=True)
    for i in range(len(g[0].edges())):
        d += edges[i][2]["weight"]
    return d + w


if __name__ == '__main__':
    g = read_graph("Heuristic/instance001.gr")

