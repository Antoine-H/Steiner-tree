
#
#  simulated anhiling.
#

import networkx as nx
import random
import local_search as ls 
import parser
import matplotlib.pyplot as plt

def local_search_only_better(nb_step  = 10):
	cur_sol = (ls.first_solution(graph, terminals))
	act_gain = ls.gain(cur_sol)
	l_act = [act_gain]
	l_new = [act_gain]
	for i in range(nb_step):
		new_sol = ls.neighbors_of_solution(graph, cur_sol, terminals)
		new_gain = ls.gain(new_sol)
		l_new.append(new_gain)
		if new_gain<act_gain:
			act_gain = new_gain
			cur_sol = new_sol
		l_act.append(act_gain)
	return(l_act, l_new)



def local_search_accept_error(nb_step  = 10,p = .2):
	cur_sol = (ls.first_solution(graph, terminals))
	act_gain = ls.gain(cur_sol)
	l_act = [act_gain]
	l_new = [act_gain]
	for i in range(nb_step):
		new_sol = ls.neighbors_of_solution(graph, cur_sol, terminals)
		new_gain = ls.gain(new_sol)
		l_new.append(new_gain)
		if new_gain<act_gain:
			r = random.random()
			if r<p:
				act_gain = new_gain
				cur_sol = new_sol
		l_act.append(act_gain)
	return(l_act, l_new)





if __name__ == '__main__':
	g = parser.read_graph("Heuristic/instance039.gr")
	graph = g[0]
	terminals = g[1]
	sol = local_search_dummy(300)
	plt.plot(sol[0],'ro')
	plt.plot(sol[1],'b^')
	plt.show()
	print(sol[0])
	print(sol[1])
	