import argparse
import copy
import heapq
import random
import time

import networkx as nx
import pandas as pd
import matplotlib.colors as mcolors  # For vizualization.
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout


# -------------------------------------------------------------------------------------#
#                               PARSING INPUT DATA                                    #
# -------------------------------------------------------------------------------------#

# Function for reading the input.
# The input is in 2 spreadsheets one with links and one with nodes.
def parseInput(filepath):
    # We create a dataframe reading the input in a spreadsheed form.
    nodes_dataframe = pd.read_excel(filepath, sheet_name="nodes").to_numpy()
    edges_dataframe = pd.read_excel(filepath, sheet_name="links").to_numpy()

    house_nodes = []  # Initializing the list of nodes type house.
    manhole_nodes = []  # Initializing the list of nodes type manhole.
    splitter_nodes = []  # Initializing the list of nodes type splitte.
    edges = []  # Initializing the list of edges.

    # Reading the data from the spreadsheed and adding the nodes to
    # their corresponding list, based on the type.
    for row in nodes_dataframe:
        node_type = row[1]
        if node_type == "PJ":
            manhole_nodes.append(row[0])
        if node_type == "TS":
            splitter_nodes.append(row[0])
        if node_type == "OS":
            house_nodes.append(row[0])

    # Adding the edges from the spreadshit to the list.
    for row in edges_dataframe:
        edges.append((row[1], row[2], row[3]))

    # Returning all four lists.
    return house_nodes, manhole_nodes, splitter_nodes, edges


# Function that creates a graph.
def createGraph(house_nodes, manhole_nodes, splitter_nodes, edges):
    G = nx.Graph()  # Creating an empty networkx graph.

    # Adding all of the nodes.
    G.add_nodes_from(house_nodes)
    G.add_nodes_from(manhole_nodes)
    G.add_nodes_from(splitter_nodes)

    # Adding all of the edges.
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    return G


# Function for reading the input.
# The input is in 2 spreadsheets one with links and one with nodes.
def parseInput(filepath):
    # We create a dataframe reading the input in a spreadsheed form.
    nodes_dataframe = pd.read_excel(filepath, sheet_name="nodes").to_numpy()
    edges_dataframe = pd.read_excel(filepath, sheet_name="links").to_numpy()

    house_nodes = []  # Initializing the list of nodes type house.
    manhole_nodes = []  # Initializing the list of nodes type manhole.
    splitter_nodes = []  # Initializing the list of nodes type splitte.
    edges = []  # Initializing the list of edges.

    # Reading the data from the spreadsheed and adding the nodes to
    # their corresponding list, based on the type.
    for row in nodes_dataframe:
        node_type = row[1]
        if node_type == "PJ":
            manhole_nodes.append(row[0])
        if node_type == "TS":
            splitter_nodes.append(row[0])
        if node_type == "OS":
            house_nodes.append(row[0])

    # Adding the edges from the spreadshit to the list.
    for row in edges_dataframe:
        edges.append((row[1], row[2], row[3]))

    # Returning all four lists.
    return house_nodes, manhole_nodes, splitter_nodes, edges


# Function that creates a graph.
def createGraph(house_nodes, manhole_nodes, splitter_nodes, edges):
    G = nx.Graph()  # Creating an empty networkx graph.

    # Adding all of the nodes.
    G.add_nodes_from(house_nodes)
    G.add_nodes_from(manhole_nodes)
    G.add_nodes_from(splitter_nodes)

    # Adding all of the edges.
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    return G


# this method repeats the greedy algorithm until it finds a valid solution
# throws errors if no manholes are passed in, or if there are isolated nodes
# prints a message and returns an empty list if a solution could not be found
# in the # of iterations
# num_iterations defaults to -1, running until it stops
def find_disjoint_set(G, manholes, splitters, random_heuristic, num_iterations=-1):
    if len(manholes) == 0:
        raise ValueError("No manholes passed in")

    if len(list(nx.isolates(G))) != 0:
        raise ValueError("Isolated nodes in graph")

    G_copy = copy.deepcopy(G)
    sets = remove_weight_greedy(G_copy, manholes, random_heuristic)

    if num_iterations == -1:
        while not is_valid_disjoint(G_copy, sets, splitters):
            G_copy = copy.deepcopy(G)
            sets = remove_weight_greedy(G_copy, manholes, random_heuristic)
    else:
        for i in range(num_iterations - 1):
            G_copy = copy.deepcopy(G)
            sets = remove_weight_greedy(G_copy, manholes, random_heuristic)

            if is_valid_disjoint(G_copy, sets, splitters):
                break

    if not is_valid_disjoint(G_copy, sets, splitters):
        return []

    return sets


# this method evaluates a disjoint set for validity:
# - paths cannot end on a splitter
# - all edges must be fully used
# easy to add additional conditions
def is_valid_disjoint(G_copy, disjoint_sets, splitters):
    if G_copy.size(weight="weight") > 0:
        return False

    for path in disjoint_sets:
        if path[-1] in splitters:
            return False

    return True


# this method iterates through a graph, removing weight as it goes
# it utilizes a heuristic to guide its path
def remove_weight_greedy(G_copy, manholes, random_heuristic):
    disjoint_sets = []

    manhole_weights = sum_manhole_weights(G_copy, manholes)

    while sum(manhole_weights.values()) > 0:

        path = [random.choice(list(manhole_weights.keys()))]

        while len(G_copy.adj[path[len(path) - 1]]) > 0:
            bestNode = get_best_node(G_copy, path, random_heuristic)

            if bestNode == -1:
                break

            path.append(bestNode)

        disjoint_sets.append(path)

        manhole_weights = sum_manhole_weights(G_copy, manholes)

    return disjoint_sets


# this method gets the sum edge weights of each manhole
def sum_manhole_weights(G_copy, manholes):
    manhole_weights = {}

    for manhole in manholes:
        weight = sum([G_copy[manhole][x]["weight"] for x in G_copy.adj[manhole]])
        if weight > 0:
            manhole_weights[manhole] = weight

    return manhole_weights


# this method identifies the best node by using a given heuristic
def get_best_node(G_copy, path, random_heuristic):
    start = path[len(path) - 1]
    adj = G_copy.adj[start]
    queue = []

    for end in adj.keys():
        if G_copy[start][end]["weight"] > 0 and end not in path:
            if random_heuristic:
                heapq.heappush(queue, (random.random(), end))
            else:
                heapq.heappush(queue, (heuristic(G_copy, start, end), end))

    if len(queue) == 0:
        return -1

    best_node = queue.pop()[1]
    G_copy[start][best_node]["weight"] = G_copy[start][best_node]["weight"] - 1
    return best_node


# this method evaluates an edge for ranking by the get_best_node method
# larger values are weighted higher
def heuristic(G_copy, start, end):
    return -1 * G_copy[start][end]["weight"]


# -------------------------------------------------------------------------------------#
#                               GRAPH VIZUALIZATION                                   #
# -------------------------------------------------------------------------------------#

# A helper function to darken the colour of the edges, so the vizualization
# looks more readable.
def darken_color(color, amount=0.8):  # amount is percantage.
    rgb = mcolors.hex2color(color)  # Changing the format.
    hsv = mcolors.rgb_to_hsv(rgb)  # Changing the format again.
    new_value = hsv[2] * amount  # Darkening the colour.
    new_rgb = mcolors.hsv_to_rgb((hsv[0], hsv[1], new_value))  # Changing back the format.
    new_hex = mcolors.rgb2hex(new_rgb)  # Changing back the format again.
    return new_hex


# A helper function for adding the edges to the multigraph with colors and edge angles.
def add_edge(G, a, b, edgeColor=0):
    # If the edge is already in the graph we must ad an angle to the edge, so there is no overlap.
    if (a, b) in G.edges:
        max_rad = max(x[2]['rad'] for x in G.edges(data=True) if sorted(x[:2]) == sorted([a, b]))
    else:  # If the edge isn't in the graph it can be straight.
        max_rad = 0
    G.add_edge(a, b, rad=max_rad + 0.1, color=edgeColor)  # Adding the edge with angle and colour.


# Creates a multigraph for the vizualization of the solution.
def createMultigraph(house_nodes, manhole_nodes, splitter_nodes, paths):
    G_multi = nx.MultiDiGraph()  # Creates an empty graph.
    G_multi.add_nodes_from(house_nodes)
    G_multi.add_nodes_from(manhole_nodes)
    G_multi.add_nodes_from(splitter_nodes)

    # Creating random colours so we can use them for showing paths.
    valid_colors = list(mcolors.CSS4_COLORS.values())  # List of valid color names
    random.shuffle(valid_colors)

    # Iterating through pairs of colours and paths, and adding them to the graph.
    for color, path in zip(valid_colors[:len(paths)], paths):
        for i in range(len(path) - 1):  # Going over each edge in the path.
            add_edge(G_multi, path[i], path[i + 1], darken_color(color))
    return G_multi


# Function that vizualizes the graph.
# The specifications are for prettier output.
def visualizeGraph(G, pos, house_nodes, manhole_nodes, splitter_nodes, weights, ax=None):
    nx.draw_networkx_nodes(G, pos, house_nodes, node_shape='o',
                           edgecolors="black", linewidths=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, manhole_nodes, node_shape='s',
                           edgecolors="black", linewidths=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, splitter_nodes, node_shape='D',
                           edgecolors="black", linewidths=0.5, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.5, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=8,
                                 font_family="serif", rotate=False, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="serif", ax=ax)


# Function that vizualizes the solution (multigraph).
# Each colour is a disjoint path.
# The specifications are for prettier output.
def visualizeMultigraph(G_multi, pos, house_nodes, manhole_nodes, splitter_nodes, ax=None):
    nx.draw_networkx_nodes(G_multi, pos, house_nodes, node_shape='o', 
                           edgecolors="black", linewidths=0.5, ax=ax)
    nx.draw_networkx_nodes(G_multi, pos, manhole_nodes, node_shape='s', 
                           edgecolors="black", linewidths=0.5, ax=ax)
    nx.draw_networkx_nodes(G_multi, pos, splitter_nodes, node_shape='D', 
                           edgecolors="black", linewidths=0.5, ax=ax)
    for edge in G_multi.edges(data=True):
        nx.draw_networkx_edges(G_multi, pos, 
                               edgelist=[(edge[0],edge[1])],
                               connectionstyle=f'arc3, rad = {edge[2]["rad"]}', 
                               width=1.5, arrowstyle="-", 
                               edge_color=edge[2]["color"], ax=ax)
    nx.draw_networkx_labels(G_multi, pos, font_size=6, font_family="serif", ax=ax)


# -------------------------------------------------------------------------------------#
#                               NO-MANHOLE PATHS CHECK                                #
# -------------------------------------------------------------------------------------#

# Checking if the solution has paths with no manholes.
def checkPaths(paths):
    for path in paths:
        has_manhole = False
        for node in path:
            # If a path has a manhole we stop the search
            # in this path and move on to the next one.
            if "M" in node:
                has_manhole = True
                break
        # If a path doesn't have a manhole we return false,
        # therefore the solution is not satisfactory.
        if not has_manhole:
            return False
    # If all paths had at least one manhole, we return true.
    return True


# -------------------------------------------------------------------------------------#
#                                       REPORT                                        #
# -------------------------------------------------------------------------------------#

# Writing up the report of the model to the user.
def writeReport(paths, execution_time):
    report = "REPORT:\n"
    report += f"Execution time: {execution_time:.4f} seconds.\n"

    report += "\nPATHS: \n"
    for i, path in enumerate(paths):
        paths_str = ", ".join(path)
        report += f"{i:3}: {paths_str}\n"
    print(report)


# -------------------------------------------------------------------------------------#
#                              PARSING ARGUMENTS                                      #
# -------------------------------------------------------------------------------------#

# Specifying the arguments for the input of the program.
def getParser():
    parser = argparse.ArgumentParser(description="Greedy method for finding disjoint paths.")
    parser.add_argument("input_file", help="This the location of the input file.")
    parser.add_argument("-i", "--iterations_number",
                        help="Number of iterations to do.",
                        action="store",
                        default=-1)
    parser.add_argument("-v", "--visualize",
                        help="Use this flag if you want vizualization of the solution.",
                        action="store_true")

    parser.add_argument("-rh", "--random_heuristic",
                        help="Use this flag if you want a random heuristic rather than the existing one.",
                        action="store_true",
                        default=False)

    return parser

# -------------------------------------------------------------------------------------#
#                                  MAIN FUNCTION                                      #
# -------------------------------------------------------------------------------------#

# Start the timer.
start_time = time.time()

# Parsing the arguments.
args = getParser().parse_args()

# First we parse the input file provided in the arguments.
# From that we get nodes of type house, splitter and manhole.
# We also get edges and edge weights.
house_nodes, manhole_nodes, splitter_nodes, edges = parseInput(args.input_file)

# Then we create an undirected graph using these nodes and edges.
G = createGraph(house_nodes, manhole_nodes, splitter_nodes, edges)
G_directed = G.to_directed()  # We also create a directed version of the graph.
# We extract the weights vector for all edges (directed).
weights = nx.get_edge_attributes(G_directed, 'weight')

paths = find_disjoint_set(G, manhole_nodes, splitter_nodes, args.random_heuristic, int(args.iterations_number))

if len(paths) == 0:
    print("No solution was found for this model.")
else:

    # Getting the exection time.
    end_time = time.time()
    execution_time = end_time - start_time

    # Printing out the report.
    writeReport(paths, execution_time)
    
# Vizualizing the input as an undirected graph and a solution as a multigraph.
if args.visualize:
    pos=graphviz_layout(G, prog="twopi")
    plt.figure()
    visualizeGraph(G, pos, house_nodes, manhole_nodes, splitter_nodes, weights)
    plt.tight_layout()
    plt.savefig(f'input.png')
    if len(paths) != 0:
        plt.figure()
        G_multi = createMultigraph(house_nodes, manhole_nodes, splitter_nodes, paths)
        visualizeMultigraph(G_multi, pos, house_nodes, manhole_nodes, splitter_nodes)
        plt.tight_layout()
        plt.savefig(f'output.png')
    plt.show()
