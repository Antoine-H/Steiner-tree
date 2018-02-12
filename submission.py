


#
#  Parser.
#

import time
import networkx as nx
import random
import math
import sys

# Reads graphs.
def read_graph (file):
    graph     = nx.Graph()
    terminals = nx.Graph()
    with open(file) as input:
        graph.add_weighted_edges_from([(int(e.split()[1]),
                                        int(e.split()[2]),
                                        int(e.split()[3]))
                                        for e in input if e.startswith("E ")])
    with open(file) as input:
        terminals.add_nodes_from([int(u.split()[1])
                                  for u in input if u.startswith("T ")])

    return [graph,terminals]




def read_graph_stdin ():
    graph     = nx.Graph()
    terminals = nx.Graph()
    
    for e in sys.stdin:
        if e.startswith("E "):
            graph.add_weighted_edges_from([(int(e.split()[1]),int(e.split()[2]),int(e.split()[3]))])
        if e.startswith("T "):
            terminals.add_nodes_from([int(e.split()[1])])

    return [graph,terminals]


########################################################################################################################



# First solution : 2-approx
def first_solution (graph,terminals):
    graph_t = nx.Graph()
    too_add = []
    approx_spanning = nx.Graph()
    ter = terminals.nodes()
    for n1 in ter:
        for n2 in ter:
            if n1 < n2:
                w = nx.shortest_path_length(graph,n1, n2,"weight")
                too_add.append((n1,n2,w))
    graph_t.add_weighted_edges_from(too_add)
    spanning_tree = nx.minimum_spanning_tree(graph_t)
    for (i,j) in spanning_tree.edges():
        path = nx.shortest_path(graph,i, j,"weight")
        for i in range(len(path)-1):
            data = graph.get_edge_data(path[i],path[i+1])["weight"]
            approx_spanning.add_edge(path[i],path[i+1],weight=data)
    return approx_spanning


#Louis version, without indexing problem (edges can't be indexed)
def is_admissible (subgraph, terminals):
    n0   = list(terminals.nodes())[0]
    comp = nx.node_connected_component(subgraph,n0)
    for n in terminals.nodes():
        if n not in comp:
            return(False)
    return True


#test test
########################################
def test_is_admissible():
    g = parser.read_graph("Heuristic/instance001.gr")
    graph     = g[0]
    terminals = g[1]
    g0 = first_solution(graph, terminals)
    print(is_admissible(g0,terminals))
    e = list(g0.edges())
    random_edge = random.choice(e)
    g0.remove_edge(*random_edge)
    print(is_admissible(g0,terminals))
########################################


#output the list of edges that can be added
def edges_adjacent (graph, subgraph):
    edges_adj = []
    sub_nodes = subgraph.nodes()
    for n1 in sub_nodes:
        for n2 in graph.neighbors(n1):
            if not subgraph.has_edge(n1,n2):
                edges_adj.append((n1,n2))
    return edges_adj


#output the list of edges that can be removed
def edges_to_delete (subgraph, terminals):
    edges_to_del = []
    graph_copy   = subgraph.copy()
    for e in list(subgraph.edges()):
        data = subgraph.get_edge_data(*e)["weight"]
        graph_copy.remove_edge(*e)
        if(is_admissible(graph_copy,terminals)):
            edges_to_del.append(e)
        graph_copy.add_edge(*e,weight = data)
    return edges_to_del


def add_path (graph,subgraph, n1 ,n2):
    path = nx.shortest_path(graph,n1, n2,"weight")
    for i in range(len(path)-1):
        data = graph.get_edge_data(path[i],path[i+1])["weight"]
        subgraph.add_edge(path[i],path[i+1],weight=data)


def add_random_path (graph, subgraph):
    list_e = list(subgraph.nodes())
    n1     = random.choice(list_e)
    n2     = random.choice(list_e)
    if (n1!=n2):
        add_path(graph, subgraph, n1, n2)


#
def clean_composante (subgraph, terminals):
    n0   = list(terminals.nodes())[0]
    comp = nx.node_connected_component(subgraph,n0)
    l = list(subgraph.nodes())
    for n in l:
        if n not in comp:
            subgraph.remove_node(n)


#
def clean (subgraph, terminals):
    clean_composante(subgraph, terminals)
    l = list(subgraph.edges())
    random.shuffle(l)
    for e in l:
            data = subgraph.get_edge_data(*e)["weight"]
            subgraph.remove_edge(*e)
            if(is_admissible(subgraph, terminals)):
                clean(subgraph, terminals)
                break
            else:
                subgraph.add_edge(*e,weight = data)


# Adds a random edge to the current solution.
def random_add (graph, cur_sol):
    list_e = edges_adjacent(graph ,cur_sol)
    random_edge = random.choice(list_e)
    data = graph.get_edge_data(*random_edge)["weight"]
    cur_sol.add_edge(*random_edge,weight = data)


# Deletes a random edge from the current solution.
def random_deletion (cur_sol, terminals):
    list_e = edges_to_delete(cur_sol, terminals)
    if list_e != []:
        random_edge = random.choice(list_e)
        cur_sol.remove_edge(*random_edge)
        clean_composante(cur_sol, terminals)



#n = nombre de test
def nm_step_dummy (graph, cur_sol, terminals,  n=40,m=40):
    if n > 0:
        random_add(graph, cur_sol)
        return nm_step_dummy(graph, cur_sol, terminals, n-1, m)
    else:
        if m > 0:
            random_deletion( cur_sol, terminals)
            return nm_step_dummy(graph, cur_sol, terminals, n, m-1)
        else:
            return cur_sol


def one_step_search (graph, cur_sol, terminals):
    p = random.random()
    if p < 0.33:
        nm_step_dummy(graph, cur_sol, terminals, 10,0)
    else:
        if p < 0.66:
            add_random_path(graph, cur_sol)
        else:
            nm_step_dummy(graph, cur_sol, terminals, 0, 10)

def one_step_search_v2(graph, cur_sol, terminals):
        p = random.random()
        if p < 0.5:
            add_random_path(graph,cur_sol) #ajout de path
        else: 
            nm_step_dummy(graph, cur_sol, terminals, 10,0)#ajoute d'edges

def one_step_search_v3(graph, cur_sol, terminals):
        p = random.random()
        if p < 0.5:
            add_random_path(graph,cur_sol) #ajout de path
        else: 
            nm_step_dummy(graph, cur_sol, terminals, 10,0)#ajoute d'edges
        p = random.random()
        if p < 0.5:
            add_random_path(graph,cur_sol) #ajout de path
        else: 
            nm_step_dummy(graph, cur_sol, terminals, 10,0)#ajoute d'edges
        nm_step_dummy(graph, cur_sol, terminals, 0, 10)

def neighbors_of_solution (graph, cur_sol, terminals, version_number = 2, nb_modif = 10):
    act      = gain(cur_sol)
    solution = cur_sol
    for i in range(nb_modif):
        new_sol  = solution.copy()
        if version_number==2:
            one_step_search_v2(graph, new_sol, terminals)
        else:
            if version_number==3:
                one_step_search_v3(graph, new_sol, terminals)
            else:
                one_step_search(graph, new_sol, terminals)
        
        new_gain = gain(new_sol)
        solution = new_sol
        gain_act = new_gain
    clean(solution, terminals)
    #print(gain(solution))
    return solution


# Objective function
def gain (steiner):
    # Assuming that max eccentricity nodes are terminals
    w = nx.diameter(steiner)
    d = 0
    edges = steiner.edges()
    for e in edges:
        data = steiner.get_edge_data(*e)["weight"]
        d   += data
    return d + w

def real_gain (steiner):
    d = 0
    edges = steiner.edges()
    for e in edges:
        data = steiner.get_edge_data(*e)["weight"]
        d   += data
    return d 

def final_value (steiner):
    # Assuming that max eccentricity nodes are terminals
    d = 0
    edges = steiner.edges()
    for e in edges:
        data = steiner.get_edge_data(*e)["weight"]
        d   += data
    return d 

# Local search. With optional parameter p \in [0,1]
# Louis : changement de la fonction, elle renvoyait pas assez new_sol
def local_search (heuristic, graph, cur_sol, terminals, p=0.25):
    new_sol = heuristic(graph, cur_sol, terminals)
    if gain(cur_sol) > gain(new_sol):
        return new_sol
    elif random.random() < p:
        #print("choix force")
        return new_sol
    else:
        return cur_sol


def test (heuristic, graph, terminals,nb_test = 5, p=0, new=nx.Graph()):
    new = first_solution (graph, terminals)
    #print("le premier gain est : "+ str(gain(new)))
    k = 0
    while k < nb_test:
        k+=1
        new = local_search (heuristic, graph, new, terminals)
        #print("le gain actuel est : "+ str(gain(new)))
    return new





########################################################################################################################

def local_search_only_better(nb_step  = 10, version = 1):
    cur_sol  = ( first_solution(graph, terminals))
    act_gain =  gain(cur_sol)
    l_act    = [act_gain]
    l_new    = [act_gain]
    for i in range(nb_step):
        print(i)
        new_sol  =  neighbors_of_solution(graph, cur_sol, terminals,version,5)
        new_gain =  gain(new_sol)
        l_new.append(new_gain)
        if new_gain < act_gain:
            act_gain = new_gain
            cur_sol  = new_sol
        l_act.append(act_gain)
    return l_act, l_new

def local_search_only_better_graph(nb_step  = 10, version = 1):
    cur_sol  = ( first_solution(graph, terminals))
    act_gain =  gain(cur_sol)
    l_act    = [act_gain]
    l_new    = [act_gain]
    for i in range(nb_step):
        new_sol  =  neighbors_of_solution(graph, cur_sol, terminals,version,5)
        new_gain =  gain(new_sol)
        l_new.append(new_gain)
        if new_gain < act_gain:
            act_gain = new_gain
            cur_sol  = new_sol
        l_act.append(act_gain)
    return cur_sol

def local_search_accept_error(nb_step  = 10, version =2,p = .1):
    cur_sol  = ( first_solution(graph, terminals))
    act_gain =  gain(cur_sol)
    l_act    = [act_gain]
    l_new    = [act_gain]
    for i in range(nb_step):
        new_sol  =  neighbors_of_solution(graph, cur_sol, terminals)
        new_gain =  gain(new_sol)
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
    heat    = 5000/nb_step #a changer
    delta   = float(act_gain-new_gain)/act_gain
    r_seuil = math.exp(delta/heat)
    print(r_seuil, nb_step,act_gain,new_gain, act_gain-new_gain)
    return r_seuil

def test_exp():
    for i in range(10):
        print(math.exp(-i))

def simulated_anhilling(nb_step = 10, heat_strategy = heat_strategy_linear):
    cur_sol  = ( first_solution(graph, terminals))
    act_gain =  gain(cur_sol)
    l_act    = [act_gain]
    l_new    = [act_gain]
    l_seuils = [1]
    for nb_step_act in range(nb_step):
        new_sol  =  neighbors_of_solution(graph, cur_sol, terminals)
        new_gain =  gain(new_sol)
        l_new.append(new_gain)
        r       = random.random()
        r_seuil = heat_strategy(nb_step_act+1, act_gain, new_gain)
        l_seuils.append(r_seuil)
        if r < r_seuil:
            act_gain = new_gain
            cur_sol  = new_sol
        l_act.append(act_gain)
    return l_act, l_new, l_seuils


#########################################################################################################




def print_solution(solution):
    print("VALUE "+str(real_gain(solution)))
    l_e = list(solution.edges())
    for a in l_e:
        print(str(a[0])+" "+str(a[1]))


if __name__ == '__main__':
    g = read_graph_stdin()
    temps0 = time.clock()
    graph = g[0]
    terminals = g[1]
    temps1 = time.clock()
    print(temps1-temps0)
    sol = local_search_only_better_graph(100,2)
    print_solution(sol)
    temps2 = time.clock()
    print(temps2-temps1)
