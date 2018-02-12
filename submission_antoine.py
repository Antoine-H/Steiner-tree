#!/usr/bin/python3

#
#  Genetic inspired heuristic.
#

import networkx as nx
import random
import math
import signal
import sys

min_sol = []


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


# A solution is admissible if all terminals are in the same connected
# component.
def is_admissible (subgraph, terminals):
	n0	 = list(terminals.nodes())[0]
	comp = nx.node_connected_component(subgraph,n0)
	for n in terminals.nodes():
		if n not in comp:
			return(False)
	return True



# First solution : 2-approx
def first_solution (graph,terminals):
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


# Generates solutions based on local search.
def random_solution (graph, terminals, n):
	sols	  = []
	first_sol = first_solution(graph, terminals)
	for i in range(n):
		n_graph = neighbors_of_solution(
						graph, first_sol, terminals, 10)
		poids	= gain(n_graph)
		sols.append((poids, n_graph, 0))
	return sols


# Merges two graphs graph0 and graph1.
def fusion (graph0, graph1, terminals):
	new_graph = nx.compose(graph0, graph1)
	clean(new_graph, terminals)
	return new_graph


# Generates mu graphs.
def initialisation (graph, terminals, mu):
	return random_solution(graph, terminals, mu)


# Samples lda out of mu graphs based on crossover technique.
def variation_crossover (terminals, population, lda):
	for i in range(lda):
		to_merge = random.sample(population, 2)
		population.append(fusion(to_merge[0][1],
					to_merge[1][1],
					terminals))

		population[-1] = (gain(population[-1]),
				population[-1],
				to_merge[0][2] + to_merge[1][2] + 1)
	return population


# Samples lda out of mu graphs based on mutation technique.
def variation_mutation (terminals, population, lda):
	for i in range(lda):
		to_mutate = random.sample(population, 1)
		population.append(neighbors_of_solution (graph,
						to_mutate[0][1],
						terminals,
						2,
						1))

		population[-1] = (gain(population[-1]),
				population[-1],
				to_mutate[0][2] + 1)
	return population


# Samples lda out of mu graphs based on both mutation and crossover techniques.
def variation_multiple (terminals, population, lda):
	return (variation_mutation(terminals, population, lda // 2) +
	variation_crossover(terminals,
				population, (lda +1) // 2)[-(lda+1)//2:])


# Selects mu graphs out of lda + mu graphs.
# elitist := max gain
def selection_elitist_classic (population, mu, t):
	population.sort(key=lambda pop: pop[0])
	return population[:mu]


# Precondition: mu <= lda. If mu > lda, falls back to
# selection_elitist_classic.
# Selects mu graphs out of lda offsprings graphs.
def selection_elitist_offsprings (population, mu, t):
	if mu <= len(population)-mu:
		to_consider = population[-mu:]
		to_consider.sort(key=lambda pop: pop[0])
		return selection_elitist_classic(to_consider, mu, t)
	else:
		return selection_elitist_classic(population, mu, t)


# Fitness_proportional = fitness (=gain) over sum for all solution
def selection_fitness_proportional (population, mu, t):
	selected = []
	for i in range(mu):
		total_gain = sum(element[0] for element in population)
		proba_vect = [(total_gain - element[0]) / total_gain
				for element in population]
		total_gain = sum(element for element in proba_vect)
		proba_vect = [element / total_gain for element in proba_vect]
		r = random.uniform(0, 1)
		for p in proba_vect:
			if r < p:
				selected.append(
					population.pop(proba_vect.index(p)))
				r = 2
			else:
				r -= p
	selected.sort(key=lambda pop: pop[0])
	return selected


# Boltzmann := accept if e^((f(y) - f(x))/T)
# T could change over time
def selection_Boltzmann (population, mu, t):
	best = min(population[:mu], key=lambda pop: pop[0])[0]
	sol = []
	for a in population[mu:]:
		r = random.uniform(0,1)
		if a[0] < best or r < math.exp((best - a[0])/t):
			sol.append(a)
	if len(sol) < mu:
		to_add = population[:mu]
		to_add.sort(key=lambda pop: pop[0])
		to_add = to_add[:mu-len(sol)]
		sol += to_add
	sol = sol[:mu]
	sol.sort(key=lambda pop: pop[0])
	return sol


# Threshold := accepft if f(y) - f(x) > T
def selection_threshold (population, mu, t):
	best = min(population[:mu], key=lambda pop: pop[0])[0]
	sol = [a for a in population[mu:] if a[0] < (best + t)]
	if len(sol) < mu:
		to_add = population[:mu]
		to_add.sort(key=lambda pop: pop[0])
		to_add = to_add[:mu-len(sol)]
		sol += to_add
	sol = sol[:mu]
	sol.sort(key=lambda pop: pop[0])
	return sol


def genetic (graph, terminals, mu, lda, variation, selection, t, threshold=3):

	initial_solutions = initialisation(graph, terminals, mu)
	current_solutions = variation(terminals, initial_solutions, lda)
	current_solutions = selection(current_solutions, mu, t)
	min_at_time = [min(current_solutions,
			key=lambda solution: solution[0])[0]]
	iteration =1
	while iteration <= threshold:
		print(iteration)
		iteration += 1
		current_solutions = variation(terminals,
						current_solutions, lda)

		current_solutions = selection(current_solutions, mu, t)
		min_at_time.append(min(current_solutions,
					key=lambda solution: solution[0])[0])

	#return min(current_solutions, key=lambda solution: solution[0])
	return min_at_time


def signal_term_handler(signal, frame):
	print_output(min_sol)
	sys.exit(0)


def print_output (solution):
	print("VALUE", solution[0])
	for e in solution[1].edges():
		print(e[0], e[1])


def genetic_sigterm (graph, terminals, mu, lda, variation, selection, t, threshold=3):

	signal.signal(signal.SIGTERM, signal_term_handler)

	global min_sol
	initial_solutions = initialisation(graph, terminals, mu)
	current_solutions = variation(terminals, initial_solutions, lda)
	current_solutions = selection(current_solutions, mu, t)

	while True:
		current_solutions = variation(terminals,
						current_solutions, lda)

		current_solutions = selection(current_solutions, mu, t)
		min_sol = min(current_solutions, key=lambda sol: sol[0])


if __name__ == '__main__':
	graph,terminals = read_graph_stdin()
	genetic_sigterm (graph, terminals, 30, 4, variation_mutation, selection_Boltzmann, 1000, 1000)

