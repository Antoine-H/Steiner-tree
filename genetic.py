
#
#  genetic.
#

import networkx as nx
import random
import local_search as ls 
import parser


def random_solution(graph, terminals, n):
	ans = []	
	first_sol= ls.first_solution(graph, terminals)
	print("on cree les premieres solution")
	for i in range(n):
		print(i)
		n_graph =ls.neighbors_of_solution(graph,first_sol, terminals,10)
		poids = ls.gain(n_graph)
		ans.append((poids, n_graph))
	print("solutions crees")
	return(ans)


def fusion(t0,t1, terminals): #juste fusion des deux graphs
	t_new = nx.compose(t0,t1)
	ls.clean(t_new, terminals)
	return(t_new)
 

def genetic(graph,terminals, nb_generation,taille_population):
	population = random_solution(graph,terminals, taille_population)
	population.sort()
	population.reverse()
	for i in range(nb_generation):
		for j in range(taille_population/2):
			population.pop()
		for j in range(taille_population/4):
			t1 = population[taille_population/4 +j]
			t0 = population[j]
			t_new = fusion(t0,t1)
			population.append(t_new)
		population_a_ajouter = random_solution(graph,terminals, taille_population/4)
		population+=population_a_ajouter
	population.sort()
	return(population[0])


if __name__ == '__main__':
	g = parser.read_graph("Heuristic/instance039.gr")
	graph = g[0]
	terminals = g[1]
	print(ls.gain(ls.first_solution(graph, terminals)))
	print(ls.gain(genetic(graph, terminals,2,8)))