
import random
import networkx as nx
import parser
import random
import matplotlib.pyplot as plt


# Displays an image of the graph.
def display (graph,name_of_graph):
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos,node_size=30)
    nx.draw_networkx_edges(graph,pos,width=5,alpha=0.5)
    plt.axis('off')
    plt.savefig(name_of_graph)
    plt.show()


# First solution : 2-approximation.
# Creates a graph consisting of shortest paths between every pair of terminal.
# Returns the minimum spanning tree of this graph.
def first_solution (graph,terminals):
    graph_t = nx.Graph()
    too_add = []
    approx_spanning = nx.Graph()
    ter = terminals.nodes()
    for n1 in ter:
        for n2 in ter:
            if n1 < n2:
                w = nx.shortest_path_length(graph, n1, n2,"weight")
                too_add.append((n1, n2, w))
    graph_t.add_weighted_edges_from(too_add)
    spanning_tree = nx.minimum_spanning_tree(graph_t)
    for (i,j) in spanning_tree.edges():
        path = nx.shortest_path(graph, i, j, "weight")
        for i in range(len(path)-1):
            data = graph.get_edge_data(path[i], path[i+1])["weight"]
            approx_spanning.add_edge(path[i], path[i+1], weight=data)
    return approx_spanning


# A solution is admissible when all the terminals are in the same connected
# component.
def is_admissible (cur_sol, terminals):
    n0   = list(terminals.nodes())[0]
    comp = nx.node_connected_component(cur_sol, n0)
    for n in terminals.nodes():
        if n not in comp:
            return(False)
    return True


# An edge can be added when one of its end is in the current solution and the
# other isn't.
def edges_adjacent (graph, cur_sol):
    edges_adj = []
    sub_nodes = cur_sol.nodes()
    for n1 in sub_nodes:
        for n2 in graph.neighbors(n1):
            if not cur_sol.has_edge(n1, n2):
                edges_adj.append((n1,n2))
    return edges_adj


# Edges can be deleted when the solution stays admissible.
def edges_to_delete (cur_sol, terminals):
    edges_to_del = []
    graph_copy   = cur_sol.copy()
    for e in list(cur_sol.edges()):
        data = cur_sol.get_edge_data(*e)["weight"]
        graph_copy.remove_edge(*e)
        if is_admissible(graph_copy, terminals):
            edges_to_del.append(e)
        graph_copy.add_edge(*e, weight = data)
    return edges_to_del


# Adds a shortest path between node u and v.
def add_path (graph, cur_sol, u ,v):
    path = nx.shortest_path(graph, u, v, "weight")
    for i in range(len(path)-1):
        data = graph.get_edge_data(path[i], path[i+1])["weight"]
        cur_sol.add_edge(path[i],path[i+1], weight=data)


# Adds a path between two random nodes.
def add_random_path (graph, cur_sol):
    list_e = list(cur_sol.nodes())
    u      = random.choice(list_e)
    v      = random.choice(list_e)
    if u != v:
        add_path(graph, cur_sol, u, v)


# No need to keep the vertices outside of the connected component where all the
# terminals are.
def clean_component (cur_sol, terminals):
    u0   = list(terminals.nodes())[0]
    comp = nx.node_connected_component(cur_sol, u0)
    l = list(cur_sol.nodes())
    for n in l:
        if n not in comp:
            cur_sol.remove_node(n)


# If removing an edge keeps the solution admissible, remove it.
def clean (cur_sol, terminals):
    clean_component(cur_sol, terminals)
    l = list(cur_sol.edges())
    random.shuffle(l)
    for e in l:
        data = graph.get_edge_data(*e)["weight"]
        cur_sol.remove_edge(*e)
        if is_admissible(cur_sol, terminals):
            clean(cur_sol, terminals)
            break
        else:
            cur_sol.add_edge(*e, weight=data)


# Adds a random edge to the current solution.
def random_add (graph, cur_sol):
    list_e = edges_adjacent(graph, cur_sol)
    random_edge = random.choice(list_e)
    data = graph.get_edge_data(*random_edge)["weight"]
    cur_sol.add_edge(*random_edge, weight=data)


# Deletes a random edge from the current solution.
def random_deletion (cur_sol, terminals):
    list_e = edges_to_delete(cur_sol, terminals)
    if list_e != []:
        random_edge = random.choice(list_e)
        cur_sol.remove_edge(*random_edge)
        clean_component(cur_sol, terminals)


# Adds n edges, removes m edges.
def nm_step_dummy (graph, cur_sol, terminals, n=40, m=40):
    if n > 0:
        random_add(graph, cur_sol)
        return nm_step_dummy(graph, cur_sol, terminals, n-1, m)
    else:
        if m > 0:
            random_deletion(cur_sol, terminals)
            return nm_step_dummy(graph, cur_sol, terminals, n, m-1)
        else:
            return cur_sol


# Performs a dummy step or adds a whole path.
def one_step_search (graph, cur_sol, terminals):
    p = random.random()
    if p < 0.33:
        nm_step_dummy(graph, cur_sol, terminals, 10, 0)
    else:
        if p < 0.66:
            add_random_path(graph, cur_sol)
        else:
            nm_step_dummy(graph, cur_sol, terminals, 0, 10)


# Performs multiple one_step_search.
def neighbors_of_solution (graph, cur_sol, terminals, nb_modifs=10):
    act      = gain(cur_sol)
    solution = cur_sol
    for i in range(nb_modifs):
        new_sol  = solution.copy()
        one_step_search(graph, new_sol, terminals)
        new_gain = gain(new_sol)
        solution = new_sol
        gain_act = new_gain
    clean(solution, terminals)
    #print(gain(solution))
    return solution


# Objective function: diameter + sum of weights of all edges.
def gain (steiner):
    # Assuming that max eccentricity nodes are terminals
    w = nx.diameter(steiner)
    d = 0
    edges = steiner.edges()
    for e in edges:
        data = steiner.get_edge_data(*e)["weight"]
        d   += data
    return d + w


# Local search. With optional parameter p \in [0,1]
def local_search (heuristic, graph, cur_sol, terminals, p=0.25):
    new_sol = heuristic(graph, cur_sol, terminals)
    if gain(cur_sol) > gain(new_sol):
        return new_sol
    elif random.random() < p:
        return new_sol
    else:
        return cur_sol


#test test
########################################
def test_is_admissible():
    g = parser.read_graph("Heuristic/instance001.gr")
    graph     = g[0]
    terminals = g[1]
    g0 = first_solution(graph, terminals)
    print(is_admissible(g0, terminals))
    e = list(g0.edges())
    random_edge = random.choice(e)
    g0.remove_edge(*random_edge)
    print(is_admissible(g0, terminals))
########################################


def test (heuristic, graph, terminals, nb_tests=5, p=0, cur_sol=nx.Graph()):
    cur_sol = first_solution (graph, terminals)
    #print("Le premier gain est :", str(gain(cur_sol)))
    k = 0
    while k < nb_tests:
        k  += 1
        cur_sol = local_search (heuristic, graph, cur_sol, terminals)
        #print("Le gain actuel est :", str(gain(cur_sol)))
    return cur_sol


if __name__ == '__main__':

    g         = parser.read_graph("Heuristic/instance039.gr")
    graph     = g[0]
    terminals = g[1]
    first_sol = first_solution(graph, terminals)

    print("Premiere valeur : ", str(gain(first_sol)))

    for i in range(30):
        neighbors_of_solution(graph, first_sol, terminals, 10)

