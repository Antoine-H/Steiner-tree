
#
#  Genetic inspired heuristic.
#

import networkx as nx
import random
import local_search as ls
import parser


# Generates solutions based on local search.
def random_solution(graph, terminals, n):
	sols      = []
	first_sol = ls.first_solution(graph, terminals)
	for i in range(n):
		n_graph = ls.neighbors_of_solution(
						graph, first_sol, terminals, 10)
		poids   = ls.gain(n_graph)
		sols.append((poids, n_graph, 0))
	return sols


# Merges two graphs g0 and g1.
def fusion(graph0, graph1, terminals):
	new_graph = nx.compose(graph0, graph1)
	ls.clean(new_graph, terminals)
	return new_graph


#
def genetic(graph, terminals, nb_generation, taille_population):
	population = random_solution(graph, terminals, taille_population)
	population.sort(key=lambda pop: pop[0], reverse=True)
	#population.sort()
	#population.reverse()
	for i in range(nb_generation):
		print("On attaque la generation numero", str(i))
		del population[-taille_population // 2:]
		#for j in range(taille_population/2):
		#	population.pop()
		for j in range(taille_population // 4):
			##fusions
			occurence1 = population[taille_population // 4 + j][2]
			occurence0 = population[j][2]
			t1 = population[taille_population // 4 + j][1]
			t0 = population[j][1]
			t_new = fusion(t0, t1, terminals)
			w = ls.gain(t_new)
			population.append((w, t_new, occurence0+occurence1 + 1))
		population_a_ajouter = random_solution( graph,
							terminals,
							taille_population // 4)
		population += population_a_ajouter
	population.sort()
	return((population[0][1], population[0][2]))


if __name__ == '__main__':
	g = parser.read_graph("Heuristic/instance039.gr")
	graph     = g[0]
	terminals = g[1]
	print(ls.gain(ls.first_solution(graph, terminals)))
	nb_gene = []
	for i in range(10):
			nb_gene_act = (i+1)*5
			print(nb_gene_act)
			nb_fusions = []
			better_res = []
			for j in range(10):
				my_sol =genetic(graph, terminals,nb_gene_act,16)
				nb_fusions.append(my_sol[1])
				better_res.append(ls.gain(my_sol[0]))
			print(nb_fusions)
			print(better_res)

