
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
	print("on cree les solutions aleatoires")
	for i in range(n):
		print(i)
		n_graph =ls.neighbors_of_solution(graph,first_sol, terminals,10)
		poids = ls.gain(n_graph)
		ans.append((poids, n_graph,0))
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
		print("on attaque la generation numero " + str(i))
		for j in range(taille_population/2):
			population.pop()
		for j in range(taille_population/4):
			occurence1 = population[taille_population/4 +j][2]
			occurence0 = population[j][2]
			t1 = population[taille_population/4 +j][1]
			t0 = population[j][1]
			t_new = fusion(t0,t1,terminals)
			w = ls.gain(t_new)
			population.append((w,t_new,occurence0+occurence1+1))
		population_a_ajouter = random_solution(graph,terminals, taille_population/4)
		population+=population_a_ajouter
	population.sort()
	return((population[0][1],population[0][2]))


if __name__ == '__main__':
	g = parser.read_graph("Heuristic/instance039.gr")
	graph = g[0]
	terminals = g[1]
	print(ls.gain(ls.first_solution(graph, terminals)))
	my_sol = genetic(graph, terminals,20,24)
	last_graph = my_sol[0]
	print("on a fait "+str(my_sol[1])+" fusions pour arriver a la meilleure solution")
	print(ls.gain(my_sol[0]))