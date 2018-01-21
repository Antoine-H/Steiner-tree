
#
#  Parser.
#

# Reads graphs.
def read_graph (file):
    graph     = []
    terminals = []
    with open(file) as input:
        for line in input:
            if line.startswith("E "):
                graph.append([int(n) for n in line.split()[1:]])
            elif line.startswith("T "):
                terminals.append([int(n) for n in line.split()[1:]])
    return [graph,terminals]

