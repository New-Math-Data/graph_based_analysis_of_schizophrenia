# Databricks notebook source
!pip install networkx

# COMMAND ----------

#https://github.com/ufvceiec/EEGRAPH/blob/481d3fe60115a1c6141c4e466144b08b609bfa6c/eegraph/tools.py#L571

pos = {

    'Fp1': (-3.15, 6.85),

    'F7': (-8.10, 4.17),
    'F3': (-4.05, 3.83), 
    'Fz': (0, 3.6),
    'F4': (4.05, 3.83),
    'F8': (8.10, 4.17),
    'T3': (-10.1,0), 
    'C3': (-5,0),
    'Cz': (0,0),
    'C4': (5,0),
    'T4': (10.1,0), 
    'T5': (-8.10, -4.17), 
    'P3': (-4.05, -3.83),
    'Pz': (0, -3.6),
    'P4': (4.05, -3.83),
    'T6': (8.10, -4.17),
    'O1': (-3.15, -6.85),
    'O2': (3.15, -6.85),
    'Fp2': (3.15, 6.85)
}
    
display(pos["Fp1"])

# COMMAND ----------

import matplotlib.pyplot as plt
import networkx as nx

# Create a new graph
G = nx.Graph()

# Add nodes with positions
for node, position in pos.items():
    G.add_node(node, pos=position)

# Add edges between nodes (optional, depending on your graph's needs)
# Example: G.add_edge('Cz', 'C2h')
# Add your edges here based on your graph's structure

# Draw the graph
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC scratch below for testing.  can probably be deleted during code cleanup.

# COMMAND ----------

import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()  # create graph object

pos = nx.circular_layout(G)  

# define list of nodes (node IDs)
nodes = [1, 2, 3, 4, 5]

# define list of edges
# list of tuples, each tuple represents an edge
# tuple (id_1, id_2) means that id_1 and id_2 nodes are connected with an edge
edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (5, 5)]

# add information to the graph object
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# draw a graph and show the plot
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

# COMMAND ----------


import numpy as np

G = nx.path_graph(20)  # An example graph
center_node = 5  # Or any other node to be in the center
edge_nodes = set(G) - {center_node}
# Ensures the nodes around the circle are evenly distributed
pos = nx.circular_layout(G.subgraph(edge_nodes))
pos[center_node] = np.array([0, 0])  # manually specify node position
nx.draw(G, pos, with_labels=True)

# COMMAND ----------


