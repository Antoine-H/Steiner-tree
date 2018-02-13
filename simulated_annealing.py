
#
#  simulated anhiling.
#
import time
import networkx as nx
import random
import local_search as ls
import parser
import matplotlib.pyplot as plt
import math

def local_search_only_better(nb_step  = 10, version = 2):
	cur_sol  = (ls.first_solution(graph, terminals))
	act_gain = ls.gain(cur_sol)
	l_act	 = [act_gain]
	l_new	 = [act_gain]
	for i in range(nb_step):
		print(i)
		new_sol  = ls.neighbors_of_solution(graph,
						cur_sol, terminals,version,5)
		new_gain = ls.gain(new_sol)
		l_new.append(new_gain)
		if new_gain < act_gain:
			act_gain = new_gain
			cur_sol  = new_sol
		l_act.append(act_gain)

	print(ls.final_value(cur_sol))
	print(ls.gain(cur_sol))
	return l_act, l_new


def local_search_accept_error(nb_step  = 10, version =2,p = .1):
	cur_sol  = (ls.first_solution(graph, terminals))
	act_gain = ls.gain(cur_sol)
	l_act	 = [act_gain]
	l_new	 = [act_gain]
	for i in range(nb_step):
		print(i)
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
	heat	= 5000/nb_step #a changer
	delta	= float(act_gain-new_gain)/act_gain
	r_seuil = math.exp(delta/heat)
	print(r_seuil, nb_step,act_gain,new_gain, act_gain-new_gain)
	return r_seuil


def simulated_anhilling(nb_step = 10, heat_strategy = heat_strategy_linear):
	cur_sol  = (ls.first_solution(graph, terminals))
	act_gain = ls.gain(cur_sol)
	l_act	 = [act_gain]
	l_new	 = [act_gain]
	l_seuils = [1]
	for nb_step_act in range(nb_step):
		new_sol  = ls.neighbors_of_solution(graph, cur_sol, terminals)
		new_gain = ls.gain(new_sol)
		l_new.append(new_gain)
		r		= random.random()
		r_seuil = heat_strategy(nb_step_act+1, act_gain, new_gain)
		l_seuils.append(r_seuil)
		if r < r_seuil:
			act_gain = new_gain
			cur_sol  = new_sol
		l_act.append(act_gain)
	return l_act, l_new, l_seuils



if __name__ == '__main__':
	g = parser.read_graph("Heuristic/instance015.gr")
	graph	  = g[0]
	terminals = g[1]
	sol1 = local_search_accept_error(1000,2, 0.01)
	sol2 = local_search_accept_error(1000,2,0.005)

	plt.plot(sol1[0],'ro')
	plt.plot(sol1[1],'b^')
	plt.show()

	plt.plot(sol2[0],'ro')
	plt.plot(sol2[1],'b^')
	plt.show()

	sol3 = local_search_accept_error(1000,2,0.001)

	plt.plot(sol3[0],'ro')
	plt.plot(sol3[1],'b^')
	plt.show()
