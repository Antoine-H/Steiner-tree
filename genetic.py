
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
		ans.append(ls.nm_step_dummy_louis(graph,first_sol, terminals, 5,5))
	print("solutions crees")
	return(ans)


def fusion(t0,t1): #juste fusion des deux graphs
	t_new = nx.compose(t0,t1)
	return(t_new)
 
def fusion_v2(t0,t1): #je sais pas ce que je voulais faire a relire ?
	t_new = nx.compose(t0,t1)
	edges = random.shuffle(t_new.edges(data=True))
	for e in edges:
		t_new.remove_edge(e)
		if(not(nx.is_connected(t_new))):
			t_new.add_adge(e)
	return(t_new)


def genetic(graph,terminals, nb_generation,taille_population):
	population = random_solution(graph,terminals, taille_population)
	for i in range(nb_generation):
		#population.sort()
		print i,len(population)
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
	g = parser.read_graph("Heuristic/instance001.gr")
	graph = g[0]
	terminals = g[1]
	print(ls.gain_louis(ls.first_solution(graph, terminals)))
	print(ls.gain_louis(genetic(graph, terminals,2,8)))