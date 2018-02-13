
import random
import networkx as nx
import parser
import random
import matplotlib.pyplot as plt
import time
import sys
sys.setrecursionlimit(1000000)


def read_graph_stdin ():
	graph	  = nx.Graph()
	terminals = nx.Graph()

	for e in sys.stdin:
		if e.startswith("E "):
			graph.add_weighted_edges_from([(int(e.split()[1]),
											int(e.split()[2]),
											int(e.split()[3]))])
		if e.startswith("T "):
			terminals.add_nodes_from([int(e.split()[1])])

	return [graph,terminals]


def display_graph (graph,name_of_graph):
	pos = nx.spring_layout(graph)
	nx.draw_networkx_nodes(graph,pos,node_size=30)
	nx.draw_networkx_edges(graph,pos,width=5,alpha=0.5)
	plt.axis('off')
	plt.savefig(name_of_graph)
	plt.show()


def first_solution(graph,terminals):
	graph_t = nx.Graph()
	too_add = []
	approx_spanning = nx.Graph()
	ter = terminals.nodes()
	#n = len(list(ter))
	#i =0
	for n1 in ter:
		#print(i,n)
		#i+=1
		sh_path_l = nx.single_source_dijkstra_path_length(graph,n1)
		for n2 in ter:
			if n1 < n2:
				w = sh_path_l[n2]
				too_add.append((n1,n2,w))
	graph_t.add_weighted_edges_from(too_add)
	spanning_tree = nx.minimum_spanning_tree(graph_t)
	for (i,j) in spanning_tree.edges():
		path = nx.shortest_path(graph,i, j,"weight")
		for i in range(len(path)-1):
			data = graph.get_edge_data(path[i],path[i+1])["weight"]
			approx_spanning.add_edge(path[i],path[i+1],weight=data)
	return approx_spanning

def first_solution_non_opti(graph,terminals):
	graph_t = nx.Graph()
	too_add = []
	approx_spanning = nx.Graph()
	ter = terminals.nodes()
	for n1 in ter:
		for n2 in ter:
			if n1 < n2:
				w = nx.shortest_path_length(graph,n1, n2,"weight")
				too_add.append((n1,n2,w))
	graph_t.add_weighted_edges_from(too_add)
	spanning_tree = nx.minimum_spanning_tree(graph_t)
	for (i,j) in spanning_tree.edges():
		path = nx.shortest_path(graph,i, j,"weight")
		for i in range(len(path)-1):
			data = graph.get_edge_data(path[i],path[i+1])["weight"]
			approx_spanning.add_edge(path[i],path[i+1],weight=data)
	return approx_spanning


def is_admissible (subgraph, terminals):
	n0	 = list(terminals.nodes())[0]
	comp = nx.node_connected_component(subgraph,n0)
	for n in terminals.nodes():
		if n not in comp:
			return(False)
	return True


#test test
########################################
def test_is_admissible():
	g = parser.read_graph("Heuristic/instance001.gr")
	graph	  = g[0]
	terminals = g[1]
	g0 = first_solution(graph, terminals)
	print(is_admissible(g0,terminals))
	e = list(g0.edges())
	random_edge = random.choice(e)
	g0.remove_edge(*random_edge)
	print(is_admissible(g0,terminals))
########################################


# Outputs the list of edges that can be added.
def edges_adjacent (graph, subgraph):
	edges_adj = []
	sub_nodes = subgraph.nodes()
	for n1 in sub_nodes:
		for n2 in graph.neighbors(n1):
			if not subgraph.has_edge(n1,n2):
				edges_adj.append((n1,n2))
	return edges_adj


# Outputs the list of edges that can be removed.
def edges_to_delete (subgraph, terminals):
	edges_to_del = []
	graph_copy	 = subgraph.copy()
	for e in list(subgraph.edges()):
		data = subgraph.get_edge_data(*e)["weight"]
		graph_copy.remove_edge(*e)
		if(is_admissible(graph_copy,terminals)):
			edges_to_del.append(e)
		graph_copy.add_edge(*e,weight = data)
	return edges_to_del


# Adds a shortest path between two given nodes in the current solution.
def add_path (graph,subgraph, n1 ,n2):
	path = nx.shortest_path(graph,n1, n2,"weight")
	for i in range(len(path)-1):
		data = graph.get_edge_data(path[i],path[i+1])["weight"]
		subgraph.add_edge(path[i],path[i+1],weight=data)


# Adds a shortest path between two random nodes in the current solution.
def add_random_path (graph, subgraph):
	list_e = list(subgraph.nodes())
	n1	   = random.choice(list_e)
	n2	   = random.choice(list_e)
	if (n1!=n2):
		add_path(graph, subgraph, n1, n2)


# Remove unnecessary edges.
# Edges that do not disconnect the solution.
def clean_composante (subgraph, terminals):
	n0	 = list(terminals.nodes())[0]
	comp = nx.node_connected_component(subgraph,n0)
	l = list(subgraph.nodes())
	for n in l:
		if n not in comp:
			subgraph.remove_node(n)


# Remove edges as long as the solution stays admissible.
def clean (subgraph, terminals):
	clean_composante(subgraph, terminals)
	l = list(subgraph.edges())
	random.shuffle(l)
	for e in l:
			data = subgraph.get_edge_data(*e)["weight"]
			subgraph.remove_edge(*e)
			if(is_admissible(subgraph, terminals)):
				clean(subgraph, terminals)
				break
			else:
				subgraph.add_edge(*e,weight = data)


# Adds a random edge to the current solution.
def random_add (graph, cur_sol):
	list_e = edges_adjacent(graph ,cur_sol)
	random_edge = random.choice(list_e)
	data = graph.get_edge_data(*random_edge)["weight"]
	cur_sol.add_edge(*random_edge,weight = data)


# Deletes a random edge from the current solution.
def random_deletion (cur_sol, terminals):
	list_e = edges_to_delete(cur_sol, terminals)
	if list_e != []:
		random_edge = random.choice(list_e)
		cur_sol.remove_edge(*random_edge)
		clean_composante(cur_sol, terminals)



# Adds n edges, removes m edges.
def nm_step_dummy (graph, cur_sol, terminals,  n=40,m=40):
	if n > 0:
		random_add(graph, cur_sol)
		return nm_step_dummy(graph, cur_sol, terminals, n-1, m)
	else:
		if m > 0:
			random_deletion( cur_sol, terminals)
			return nm_step_dummy(graph, cur_sol, terminals, n, m-1)
		else:
			return cur_sol


def one_step_search (graph, cur_sol, terminals):
	p = random.random()
	if p < 0.33:
		nm_step_dummy(graph, cur_sol, terminals, 10,0)
	else:
		if p < 0.66:
			add_random_path(graph, cur_sol)
		else:
			nm_step_dummy(graph, cur_sol, terminals, 0, 10)


def one_step_search_v2(graph, cur_sol, terminals):
		p = random.random()
		if p < 0.5:
			add_random_path(graph,cur_sol) #ajout de path
		else:
			nm_step_dummy(graph, cur_sol, terminals, 10,0)#ajoute d'edges


def one_step_search_v3(graph, cur_sol, terminals):
		p = random.random()
		if p < 0.5:
			add_random_path(graph,cur_sol) #ajout de path
		else:
			nm_step_dummy(graph, cur_sol, terminals, 10,0)#ajoute d'edges
		p = random.random()
		if p < 0.5:
			add_random_path(graph,cur_sol) #ajout de path
		else:
			nm_step_dummy(graph, cur_sol, terminals, 10,0)#ajoute d'edges
		nm_step_dummy(graph, cur_sol, terminals, 0, 10)


def neighbors_of_solution (graph, cur_sol, terminals,
										version_number = 2, nb_modif = 10):
	act		 = gain(cur_sol)
	solution = cur_sol
	for i in range(nb_modif):
		new_sol  = solution.copy()
		if version_number==2:
			one_step_search_v2(graph, new_sol, terminals)
		else:
			if version_number==3:
				one_step_search_v3(graph, new_sol, terminals)
			else:
				one_step_search(graph, new_sol, terminals)

		new_gain = gain(new_sol)
		solution = new_sol
		gain_act = new_gain
	clean(solution, terminals)
	#print(gain(solution))
	return solution


# Objective function
def gain (steiner):
	# Assuming that max eccentricity nodes are terminals
	w = nx.diameter(steiner)
	d = 0
	edges = steiner.edges()
	for e in edges:
		data = steiner.get_edge_data(*e)["weight"]
		d	+= data
	return d + w


def final_value (steiner):
	# Assuming that max eccentricity nodes are terminals
	d = 0
	edges = steiner.edges()
	for e in edges:
		data = steiner.get_edge_data(*e)["weight"]
		d	+= data
	return d

# Local search. With optional parameter p \in [0,1]
def local_search (heuristic, graph, cur_sol, terminals, p=0.25):
	new_sol = heuristic(graph, cur_sol, terminals)
	if gain(cur_sol) > gain(new_sol):
		return new_sol
	elif random.random() < p:
		#print("choix force")
		return new_sol
	else:
		return cur_sol


def test (heuristic, graph, terminals,nb_test = 5, p=0, new=nx.Graph()):
	new = first_solution (graph, terminals)
	#print("le premier gain est : "+ str(gain(new)))
	k = 0
	while k < nb_test:
		k  += 1
		new = local_search (heuristic, graph, new, terminals)
		#print("le gain actuel est : "+ str(gain(new)))
	return new


temps_debut = 0
temps_step = 0


def get_step():
	t = time.clock()
	delta = t-temps_step
	temps_step = t
	return(delta)


if __name__ == '__main__':
	g = parser.read_graph("Heuristic/instance143.gr")
	graph	  = g[0]
	print(len(graph.edges()), len(graph.nodes()))
	terminals = g[1]
	print(len(terminals))
	t0 = time.clock()
	g0 = first_solution(graph, terminals)
	t1 = time.clock()
	delta1 = t1 -t0
	print(delta1)

	g0 = first_solution_non_opti(graph, terminals)
	t2 = time.clock()
	delta2 = t2 - t1
	print(delta2)

