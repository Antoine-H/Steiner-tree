
#
#  genetic.
#

import networkx as nx
import random
import local_search as ls 
import parser

def random_solution(graph, n):
	ans = []	
	for i in range(n):
		ans.append(ls.une_solution(graph))
	return(ans)

def fusion(t0,t1):
	t_new = nx.compose(t0,t1)
	edges = random.shuffle(t_new.edges(data=True))
	for e in edges:
		t_new.remove_edge(e)
		if(not(nx.is_connected(t_new))):
			t_new.add_adge(e)
	return(t_new)


def genetic(graph,nb_generation,taille_population):
	population = random_solution(graph,taille_population)
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
	print(coucou)