
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
    print(terminals.nodes())
    for i in range(len(terminals.nodes())):
        print(i)
        for j in range(len(terminals.nodes())-i-1):
            print(j)
            graph_copy.add_weighted_edges_from(rand_edges)
            w = nx.shortest_path_length(
                    graph,terminals.nodes()[i], terminals.nodes()[j],"weight")
            too_add.append((terminals.nodes()[i],terminals.nodes()[j],w))

    graph_t.add_weighted_edges_from(too_add)
    spanning_tree = nx.minimum_spanning_tree(graph_t)

    for (i,j) in spanning_tree.edges():
        path = nx.shortest_path(graph,i, j,"weight")
        for i in range(len(path)-1):
            approx_spanning.add_edge(path[i],path[i+1])
    return approx_spanning


def display (graph):
    pos=nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos,node_size=30)
    nx.draw_networkx_edges(graph,pos,width=5,alpha=0.5)
    plt.axis('off')
    plt.savefig("weighted_graph.png")
    plt.show()


def nm_step_dummy(graph, cur_sol, terminals, n=1, m=1):

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


# Objective function
def gain (steiner):
    # Assuming that max eccentricity nodes are terminals
    w = nx.diameter(steiner)
    d = 0
    edges = g[0].edges(data=True)
    for i in range(len(g[0].edges())):
        d += edges[i][2]["weight"]
    return d + w


# Local search. With optional parameter p \in [0,1]
# Louis : changement de la fonction, elle renvoyait pas assez new_sol
def local_search (heuristic, graph, cur_sol, terminals, p=0):
    new_sol = heuristic(graph, cur_sol, terminals)
    if gain(cur_sol) > gain(new_sol):
        return new_sol
    elif random.random() < p:
        return new_sol
    else:
        return cur_sol


def test (heuristic, graph, terminals, new=nx.Graph(), p=0):
    new = first_solution (graph, terminals)
    print(gain(new))
    while True:
        new = local_search (heuristic, graph, new, terminals, p=0)
        print(gain(new))



if __name__ == '__main__':
    g = parser.read_graph("Heuristic/instance001.gr")
    f_s = first_solution(g[0],g[1])
    print(g[1].nodes())

