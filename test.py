import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt

# Create a sample graph
G = nx.karate_club_graph()

# Compute the best partition (modularity classes)
partition = community_louvain.best_partition(G)

# Draw the graph
pos = nx.spring_layout(G)
cmap = plt.get_cmap("viridis")

# Create a color map from the partition values
unique_partitions = list(set(partition.values()))
color_map = [
    cmap(unique_partitions.index(partition[node]) / len(unique_partitions))
    for node in G.nodes()
]

# Plot the graph
plt.figure(figsize=(10, 7))
nx.draw(
    G,
    pos,
    node_color=color_map,
    with_labels=True,
    node_size=300,
    font_size=10,
    font_color="white",
)
plt.title("Graph with Nodes Colored by Modularity Class")
plt.show()
