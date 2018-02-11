
#
#  simulated anhiling.
#

import networkx as nx
import random
import local_search as ls
import parser
import matplotlib.pyplot as plt
import math

def local_search_only_better(nb_step  = 10):
	cur_sol  = (ls.first_solution(graph, terminals))
	act_gain = ls.gain(cur_sol)
	l_act    = [act_gain]
	l_new    = [act_gain]
	for i in range(nb_step):
		new_sol  = ls.neighbors_of_solution(graph, cur_sol, terminals)
		new_gain = ls.gain(new_sol)
		l_new.append(new_gain)
		if new_gain < act_gain:
			act_gain = new_gain
			cur_sol  = new_sol
		l_act.append(act_gain)
	return l_act, l_new



def local_search_accept_error(nb_step  = 10,p = .1):
	cur_sol  = (ls.first_solution(graph, terminals))
	act_gain = ls.gain(cur_sol)
	l_act    = [act_gain]
	l_new    = [act_gain]
	for i in range(nb_step):
		new_sol  = ls.neighbors_of_solution(graph, cur_sol, terminals)
		new_gain = ls.gain(new_sol)
		l_new.append(new_gain)
		r = random.random()
		if new_gain<act_gain or r<p:
			act_gain = new_gain
			cur_sol  = new_sol
		l_act.append(act_gain)
	return l_act, l_new

def heat_strategy_linear(nb_step, act_gain, new_gain):
	if act_gain > new_gain:
		return 1
	heat    = 5000/nb_step
	r_seuil = math.exp(float(act_gain-new_gain)/heat)
	print(r_seuil, nb_step,act_gain,new_gain, act_gain-new_gain)
	return r_seuil

def test_exp():
	for i in range(10):
		print(math.exp(-i))

def simulated_anhilling(nb_step = 10, heat_strategy = heat_strategy_linear):
	cur_sol  = (ls.first_solution(graph, terminals))
	act_gain = ls.gain(cur_sol)
	l_act    = [act_gain]
	l_new    = [act_gain]
	l_seuils = [1]
	for nb_step_act in range(nb_step):
		new_sol  = ls.neighbors_of_solution(graph, cur_sol, terminals)
		new_gain = ls.gain(new_sol)
		l_new.append(new_gain)
		r       = random.random()
		r_seuil = heat_strategy(nb_step_act+1, act_gain, new_gain)
		l_seuils.append(r_seuil)
		if r < r_seuil:
			act_gain = new_gain
			cur_sol  = new_sol
		l_act.append(act_gain)
	return l_act, l_new, l_seuils



if __name__ == '__main__':
	g = parser.read_graph("Heuristic/instance039.gr")
	graph     = g[0]
	terminals = g[1]
	sol = local_search_accept_error(1000,0.005)
	plt.plot(sol[0],'ro')
	plt.plot(sol[1],'b^')
	#plt.plot(sol[2],'b^')
	plt.show()

