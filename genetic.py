
#
#  Parser.
#

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
 

def fusion(t0,t1):
	t_new = nx.compose(t0,t1)
	edges = shuffle(t_new.edges(data=True))
	for e in edges:
		t_new.remove_edge(e)
		if(not(nx.is_connected(t_new))):
			t_new.add_adge(e)
	return(t_new)


def genetic(nb_generation,taille_population):
	population = random_solution(taille_population)
	for i in range(nb_generation):
		population.sort()
		for j in range(taille_population/2):
			population.pop()
		for j in range(taille_population/4):
			t1 = population[taille_population/4 +j][1]
			t0 = population[j][0]
			t_new = fusion(t0,t1)
			population.append(t_new)
	population.sort()
	return(population[0])

if __name__ == '__main__':
    g = read_graph("Heuristic/instance001.gr")

