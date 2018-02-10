
import random
import networkx as nx
import parser
import random
import matplotlib.pyplot as plt

def display (graph,name_of_graph):
    pos=nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos,node_size=30)
    nx.draw_networkx_edges(graph,pos,width=5,alpha=0.5)
    plt.axis('off')
    plt.savefig(name_of_graph)
    plt.show()

data1 = []
data2 = []

# First solution : 2-approx
def first_solution(graph,terminals):
    graph_t = nx.Graph()
    too_add = []
    approx_spanning = nx.Graph() 
    ter = terminals.nodes()
    for n1 in ter:
        for n2 in ter:
            if n1<n2:
                w = nx.shortest_path_length(graph,n1, n2,"weight")
                too_add.append((n1,n2,w))
    graph_t.add_weighted_edges_from(too_add)
    spanning_tree = nx.minimum_spanning_tree(graph_t)
    for (i,j) in spanning_tree.edges():
        path = nx.shortest_path(graph,i, j,"weight")
        for i in range(len(path)-1):
            data = graph.get_edge_data(path[i],path[i+1])["weight"]
            data1.append(data)
            approx_spanning.add_edge(path[i],path[i+1],weight=data)
    return approx_spanning


#local search algorithm Antoine

def nm_step_dummy_antoine(graph, cur_sol, terminals, n=5, m=5):
    graph_copy = cur_sol.copy()
    print(nx.is_connected(graph_copy))
    rand_edges = [graph.edges(data=True)[i]
                    for i in random.sample(range(len(graph)),n)]
    print(rand_edges)
    graph_copy.add_weighted_edges_from(rand_edges)
    print(nx.is_connected(graph_copy))
    rand_edges = [graph.edges(data=True)[i]
                    for i in random.sample(range(len(graph)),m)]

    print(nx.is_connected(graph_copy))
    for e in rand_edges:
        graph_copy.remove_edges_from([e])
        if not nx.is_connected(graph_copy):
            graph_copy.add_weighted_edges_from([e])

    print(nx.is_connected(graph_copy))
    return graph_copy


#Louis version, without indexing problem (edges can't be indexed)
def is_admissible(subgraph, terminals):
    n0 = list(terminals.nodes())[0]
    comp = nx.node_connected_component(subgraph,n0)
    for n in terminals.nodes():
        if n not in comp:
            return(False)
    return(True)

#test test 
########################################
def test_is_admissible():
    g = parser.read_graph("Heuristic/instance001.gr")
    graph = g[0]
    terminals = g[1]
    g0 = first_solution(graph, terminals)
    print(is_admissible(g0,terminals))
    e = list(g0.edges())
    random_edge = random.choice(e)
    g0.remove_edge(*random_edge)
    print(is_admissible(g0,terminals))
########################################



def edges_adjacent(graph, subgraph): #output the list of edges that can be added
    edges_adj = []
    sub_nodes = subgraph.nodes()
    for n1 in sub_nodes:
        for n2 in graph.neighbors(n1): 
            if n2 not in sub_nodes:
                edges_adj.append((n1,n2))
    return(edges_adj)

def edges_to_delete(graph, terminals): #output the list of edges that can be removed
    edges_to_del  = []
    graph_copy = graph.copy()
    for e in list(graph.edges()):
        data = graph.get_edge_data(*e)["weight"]
        graph_copy.remove_edge(*e)
        if(nx.is_connected(graph_copy)):
            edges_to_del.append(e)
        graph_copy.add_edge(*e,weight = data)
    return(edges_to_del)

def add_path(graph, subgraph):
    list_e = subgraph.edges()
    n1 = random.choice(list_e)
    n2 = random.choice(list_e)
    if (n1!=n2):
        path  = nx.shortest_path_length(graph,n1, n2,"weight")
        for i in range(len(path)-1):
            data = graph.get_edge_data(path[i],path[i+1])["weight"]
            subgraph.add_edge(path[i],path[i+1],weight=data)

def clean(graph,subgraph,terminals):
    edges_to_del  = []
    for e in list(subgraph.edges()):
        data = graph.get_edge_data(*e)["weight"]
        subgraph.remove_edge(*e)
        if(nx.is_connected(graph_copy)):
            edges_to_del.append(e)
        graph_copy.add_edge(*e,weight = data)
    return(edges_to_del)    

def random_add(graph, cur_sol):
    list_e = edges_adjacent(graph ,cur_sol )
    random_edge = random.choice(list_e)
    data = graph.get_edge_data(*random_edge)["weight"]
    cur_sol.add_edge(*random_edge,weight = data)
    
def random_delation(cur_sol, terminals):
    list_e = edges_to_delete(cur_sol, terminals)
    if list_e !=[]:
        random_edge = random.choice(list_e)
        cur_sol.remove_edge(*random_edge)

def random_modif_one_step(graph, cur_sol, terminals, proba_add = 0.8, nb_fois = 1):
    graph_copy = cur_sol.copy()
    if(random.random()<proba_add):
        random_add(graph,graph_copy)
    else:
        random_delation(graph_copy, terminals)
    return(graph_copy)

def nm_step_dummy_louis(graph, cur_sol, terminals,  n=40,m=40):
    if n>0:
        random_add(graph, cur_sol)
        return(nm_step_dummy_louis(graph, cur_sol, terminals, n-1, m))
    else:
        if m>0:
            random_delation( cur_sol, terminals)
            return(nm_step_dummy_louis(graph, cur_sol, terminals, n, m-1))
        else:
            return(cur_sol)



# Objective function
def gain (steiner):
    # Assuming that max eccentricity nodes are terminals
    w = nx.diameter(steiner)
    d = 0
    edges = g[0].edges(data=True)
    for i in range(len(g[0].edges())):
        d += edges[i][2]["weight"]
    return d + w

def gain_louis (steiner):
    # Assuming that max eccentricity nodes are terminals
    #w = nx.diameter(steiner)
    d = 0
    edges = steiner.edges()
    for e in edges:
        data= steiner.get_edge_data(*e)["weight"]
        d += data
    return d 



        # Local search. With optional parameter p \in [0,1]
# Louis : changement de la fonction, elle renvoyait pas assez new_sol
def local_search (heuristic, graph, cur_sol, terminals, p=0.25):
    new_sol = heuristic(graph, cur_sol, terminals)
    if gain_louis(cur_sol) > gain_louis(new_sol):
        return new_sol
    elif random.random() < p:
        #print("choix force")
        return new_sol
    else:
        return cur_sol


def test (heuristic, graph, terminals,nb_test = 5, p=0, new=nx.Graph()):
    new = first_solution (graph, terminals)

    #print("le premier gain est : "+ str(gain_louis(new)))
    k = 0
    while k<nb_test:
        k+=1
        new = local_search (heuristic, graph, new, terminals)
        #print("le gain actuel est : "+ str(gain_louis(new)))
    return(new)


if __name__ == '__main__':
    g = parser.read_graph("Heuristic/instance001.gr")
    graph = g[0]
    terminals = g[1]
    g0 = first_solution(graph, terminals)
