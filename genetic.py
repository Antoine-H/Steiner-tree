
#
#  Genetic inspired heuristic.
#

import networkx as nx
import random
import local_search as ls
import simulated_annealing as sa
import parser
import math
import matplotlib.pyplot as plt
import signal
import sys

min_sol = []

# Generates solutions based on local search.
def random_solution (graph, terminals, n):
	sols      = []
	first_sol = ls.first_solution(graph, terminals)
	for i in range(n):
		n_graph = ls.neighbors_of_solution(
						graph, first_sol, terminals, 10)
		poids   = ls.gain(n_graph)
		sols.append((poids, n_graph, 0))
	return sols


# Merges two graphs graph0 and graph1.
def fusion (graph0, graph1, terminals):
	new_graph = nx.compose(graph0, graph1)
	ls.clean(new_graph, terminals)
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

		population[-1] = (ls.gain(population[-1]),
				population[-1],
				to_merge[0][2] + to_merge[1][2] + 1)
	return population


# Samples lda out of mu graphs based on mutation technique.
def variation_mutation (terminals, population, lda):
	for i in range(lda):
		to_mutate = random.sample(population, 1)
		population.append(ls.neighbors_of_solution (graph,
						to_mutate[0][1],
						terminals,
						2,
						1))

		population[-1] = (ls.gain(population[-1]),
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


def pretty_print(list):
	for i in list:
		print(i)


# mu + lambda evolutionary algorithm : mutation, elitist
# mu + lambda EA variant: mutation, elitist within offsprings (mu within
# lambda)
# Crossover, standard bit mutation, elitist
# Variant: wp proportional to fitness, mutation, otherwise crossover.
def genetic_v (graph, terminals, mu, lda,
				variation, selection, t, threshold=3):
	initial_solutions = initialisation(graph, terminals, mu)

	print("Initial solution:")
	pretty_print(initial_solutions)

	current_solutions = variation(terminals, initial_solutions, lda)

	print("Variations:")
	pretty_print(current_solutions)

	current_solutions = selection(current_solutions, mu, t)

	print("Selection:")
	pretty_print(current_solutions)
	print("MIN:", min(current_solutions, key=lambda solution: solution[0]))

	iteration = 1
	while iteration <= threshold:
		iteration += 1
		current_solutions = variation(terminals,
						current_solutions, lda)

		print("Variations:")
		pretty_print(current_solutions)

		current_solutions = selection(current_solutions, mu, t)

		print("Selection:")
		pretty_print(current_solutions)
		print("MIN:",
		min(current_solutions, key=lambda solution: solution[0]))

	return current_solutions


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
		print("GO")


def local_search_only_better(nb_step  = 10, version = 2):
	cur_sol  = (ls.first_solution(graph, terminals))
	act_gain = ls.gain(cur_sol)
	l_act    = [act_gain]
	l_new    = [act_gain]
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


if __name__ == '__main__':
	graph,terminals = parser.read_graph("Heuristic/instance039.gr")

	genetic_sigterm (graph, terminals, 30, 4, variation_mutation, selection_Boltzmann, 1000, 1000)
	#b = genetic (graph, terminals, 5, 2, variation_crossover, selection_threshold, -80, 1000)
	#c = genetic_no_blabla (graph, terminals, 5, 2, variation_multiple, selection_threshold, -80, 1000)
	#d = genetic_no_blabla (graph, terminals, 5, 2, variation_multiple, selection_Boltzmann, 1000, 1000)
	#e = genetic_no_blabla (graph, terminals, 5, 2, variation_multiple, selection_threshold, -150, 1000)

	#print(a)
	#print(b)
	#print(c)
	#print(d)
	#print(e)

	#plt.plot(a)
	#plt.plot(b)
	#plt.plot(c)
	#plt.plot(d)
	#plt.plot(e)

	#plt.savefig("Plots_Antoine/new/5,2,mutcmul,Threshold")
	#plt.show()

