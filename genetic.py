
#
#  Genetic inspired heuristic.
#

import networkx as nx
import random
import local_search as ls
import parser
import math
import matplotlib.pyplot as plt


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


#
def genetic_louis (graph, terminals, nb_generation, taille_population):
	population = random_solution(graph, terminals, taille_population)
	population.sort(key=lambda pop: pop[0], reverse=True)
	for i in range(nb_generation):
		print("On attaque la generation numero", str(i))
		del population[-taille_population // 2:]
		for j in range(taille_population // 4):
			##fusions
			fusions0 = population[taille_population // 4 + j][2]
			fusions1 = population[j][2]
			t1 = population[taille_population // 4 + j][1]
			t0 = population[j][1]
			t_new = fusion(t0, t1, terminals)
			w = ls.gain(t_new)
			population.append((w, t_new, fusions0 + fusions1 + 1))
		population_a_ajouter = random_solution( graph,
							terminals,
							taille_population // 4)
		population += population_a_ajouter
	population.sort(key=lambda pop: pop[0], reverse=True)
	return((population[0][1], population[0][2]))


def variation_crossover_v2 (terminals, population, lda): #LOUIS
	for i in range(lda):
		to_merge  = random.sample(population, 2)
		new_graph = (fusion(to_merge[0][1],
					to_merge[1][1],
					terminals))

		population.append((ls.gain(new_graph),
				new_graph,
				to_merge[0][2] + to_merge[1][2] + 1))
	return population


def variation_mutation_v2 (terminals, population, lda): #LOUIS
	for i in range(lda):
		to_mutate = random.sample(population, 1)
		new_graph = (ls.neighbors_of_solution (graph,
						to_mutate[0][1],
						terminals,
						2,
						1))

		population.append((ls.gain(new_graph),
				new_graph,
				to_mutate[0][2] + 1))
	return population


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
	variation_crossover(terminals, population, (lda +1) // 2)[-(lda+1)//2:])


# Selects mu graphs out of lda + mu graphs.
# elitist := max gain
def selection_elitist_classic (population, mu, t):
	population.sort(key=lambda pop: pop[0])
	return population[:mu]


# Precondition: mu <= lda. If mu > lda, falls back to selection_elitist_classic.
# Selects mu graphs out of lda offsprings graphs.
def selection_elitist_offsprings (population, mu, t):
	if mu <= lda:
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
# mu + lambda EA variant: mutation, elitist within offsprings (mu within lambda)
# Crossover, standard bit mutation, elitist
# Variant: wp proportional to fitness, mutation, otherwise crossover.
def genetic (graph, terminals, mu, lda, variation, selection, t, threshold=3):
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
		current_solutions = variation(terminals, current_solutions, lda)

		print("Variations:")
		pretty_print(current_solutions)

		current_solutions = selection(current_solutions, mu, t)

		print("Selection:")
		pretty_print(current_solutions)
		print("MIN:",
		min(current_solutions, key=lambda solution: solution[0]))

	#return min(current_solutions, key=lambda solution: solution[0])
	return current_solutions


def genetic_no_blabla (graph, terminals, mu, lda, variation,
			selection, t, threshold=3):
	initial_solutions = initialisation(graph, terminals, mu)
	current_solutions = variation(terminals, initial_solutions, lda)
	current_solutions = selection(current_solutions, mu, t)
	min_at_time = [min(current_solutions,
			key=lambda solution: solution[0])[0]]
	iteration =1
	while iteration <= threshold:
		print(iteration)
		iteration += 1
		current_solutions = variation(terminals, current_solutions, lda)

		current_solutions = selection(current_solutions, mu, t)
		min_at_time.append(min(current_solutions,
					key=lambda solution: solution[0])[0])

	#return min(current_solutions, key=lambda solution: solution[0])
	return min_at_time

if __name__ == '__main__':
	graph,terminals = parser.read_graph("Heuristic/instance039.gr")
	a = genetic_no_blabla (graph, terminals, 5, 2, variation_mutation, selection_elitist_offsprings, 1000, 10)
	#b = genetic_no_blabla (graph, terminals, 5, 2, variation_mutation, selection_elitist_offsprings, 1000, 1000)
	#c = genetic_no_blabla (graph, terminals, 5, 2, variation_mutation, selection_fitness_proportional, 1000, 1000)
	#d = genetic_no_blabla (graph, terminals, 5, 2, variation_mutation, selection_Boltzmann, 1000, 1000)
	#e = genetic_no_blabla (graph, terminals, 5, 2, variation_mutation, selection_threshold, 1000, 1000)

	print(a)
	#print(b)
	#print(c)
	#print(d)
	#print(e)

	plt.plot(a)
	#plt.plot(b)
	#plt.plot(c)
	#plt.plot(d)
	#plt.plot(e)

	plt.savefig("Plots_Antoine/a")
	plt.show()

#	lda = 2
#	mu  = 3
#	init_sols = initialisation(graph, terminals, mu)
#	cur_sols  = variation_crossover(terminals, init_sols, lda)
#	cur_sols2 = variation_mutation(terminals, cur_sols, lda)
#	sols1     = selection_elitist_classic(cur_sols2, mu, 0)
#	sols2     = selection_elitist_offsprings(cur_sols, mu, 0)
#	sols3     = selection_fitness_proportional(cur_sols2, mu, 0)
#	sols4     = selection_Boltzmann(cur_sols2, mu, 1000)
#	sols5     = selection_threshold(cur_sols2, mu, -100)
#	print(ls.gain(ls.first_solution(graph, terminals)))
#	nb_gene = []

