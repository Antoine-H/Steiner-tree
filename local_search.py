
import random
import networkx as nx
import parser
import random
import matplotlib.pyplot as plt

# Not checked
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
            approx_spanning.add_edge(path[i],path[i+1],weight=data)
    return approx_spanning


def display (graph,name_of_graph):
    pos=nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos,node_size=30)
    nx.draw_networkx_edges(graph,pos,width=5,alpha=0.5)
    plt.axis('off')
    plt.savefig(name_of_graph)
    plt.show()


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
def nm_step_dummy_louis(graph, cur_sol, terminals, n=5, m=5):
    deletions = 0
    try_deletion = 0
    graph_copy = cur_sol.copy()
    while(deletions<n and try_deletion<100):
        try_deletion +=1
        random_edge = random.choice(list(graph_copy.edges()))
        graph_copy.remove_edge(*random_edge)
        if(nx.is_connected(graph_copy)):
            deletions+=1
        else:
            data = graph.get_edge_data(*random_edge)["weight"]
            graph_copy.add_edge(*random_edge,weight = data)
    incrementation = 0
    try_incrementation = 0
    while(incrementation<m and try_incrementation<100):
        try_incrementation +=1
        random_edge = random.choice(list(graph_copy.edges()))
        data = graph.get_edge_data(*random_edge)["weight"]
        graph_copy.add_edge(*random_edge, weight = data)
        if(nx.is_connected(graph_copy)):
            incrementation+=1
        else:
            graph_copy.remove_edge(*random_edge)    
    return(graph_copy)

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
    w = nx.diameter(steiner)
    d = 0
    edges = steiner.edges()
    for e in edges:
        data= steiner.get_edge_data(*e)["weight"]
        d += data
    return d + w



        # Local search. With optional parameter p \in [0,1]
# Louis : changement de la fonction, elle renvoyait pas assez new_sol
def local_search (heuristic, graph, cur_sol, terminals, p=0.25):
    new_sol = heuristic(graph, cur_sol, terminals)
    if gain_louis(cur_sol) > gain_louis(new_sol):
        return new_sol
    elif random.random() < p:
        print("choix force")
        return new_sol
    else:
        return cur_sol


def test (heuristic, graph, terminals,nb_test = 20, p=0, new=nx.Graph()):
    new = first_solution (graph, terminals)
    print("le premier gain est : "+ str(gain_louis(new)))
    k = 0
    while k<nb_test:
        k+=1
        new = local_search (heuristic, graph, new, terminals)
        print("le gain actuel est : "+ str(gain_louis(new)))


if __name__ == '__main__':
    g = parser.read_graph("Heuristic/instance001.gr")
    test(nm_step_dummy_louis,g[0],g[1],20, 0.5)


    #local_search(nm_step_dummy_louis,g[0],f_s,g[1])
    

