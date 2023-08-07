import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import csv
import os

def create_graph(adj_matrix, config):
    G = nx.DiGraph()
    locations_path = config['locations_path']['default'] 
    with open(locations_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            number, station_name, lat, lon, province = row
            G.add_node(number, pos=(float(lon), float(lat)), province=province)
    for i in range(adj_matrix.shape[1]):  # Iterate over the columns instead of rows
        strongest_influence_indices = np.argsort(adj_matrix[i, :])[-1:]  # Enter number of influential stations
        for j in strongest_influence_indices:
            if adj_matrix[i, j] > 0:  # Use adj_matrix[i, j] instead of adj_matrix[j, i]
                G.add_edge(list(G.nodes())[j], list(G.nodes())[i])  # Swap the order of nodes
    return G

def plot_map(adj_matrix, config, split):
    hex_colors = {
        0: '#E52B50', 1: '#40826D', 2: '#8000FF', 3: '#3F00FF', 4: '#40E0D0', 5: '#008080', 6: '#483C32', 7: '#D2B48C',
        8: '#00FF7F', 9: '#A7FC00', 10: '#708090', 11: '#C0C0C0', 12: '#FF2400', 13: '#0F52BA', 14: '#92000A', 15: '#FA8072',
        16: '#E0115F', 17: '#FF007F', 18: '#C71585', 19: '#FF0000', 20: '#E30B5C', 21: '#6A0DAD', 22: '#CC8899', 23: '#003153',
        24: '#8E4585', 25: '#FFC0CB', 26: '#1C39BB', 27: '#C3CDE6', 28: '#D1E231', 29: '#FFE5B4', 30: '#DA70D6', 31: '#FF4500',
        32: '#FF6600', 33: '#808000', 34: '#CC7722', 35: '#000080', 36: '#E0B0FF', 37: '#800000', 38: '#FF00AF', 39: '#FF00FF',
        40: '#BFFF00', 41: '#C8A2C8', 42: '#FFF700', 43: '#B57EDC', 44: '#29AB87', 45: '#00A86B'
    }
    G = create_graph(adj_matrix, config)
    node_positions = nx.get_node_attributes(G, 'pos')

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    min_lon = min(pos[0] for pos in node_positions.values())
    max_lon = max(pos[0] for pos in node_positions.values())
    min_lat = min(pos[1] for pos in node_positions.values())
    max_lat = max(pos[1] for pos in node_positions.values())
    width = max_lon - min_lon
    height = max_lat - min_lat

    m = Basemap(
        llcrnrlon=min_lon - 0.1 * width, llcrnrlat=min_lat - 0.1 * height,
        urcrnrlon=max_lon + 0.1 * width, urcrnrlat=max_lat + 0.1 * height,
        resolution='i', ax=ax
    )
    m.drawcoastlines(linewidth=0.5)
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='white', lake_color='lightblue')

    node_degrees = G.out_degree()
    sorted_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)  # Sort nodes by out-degree
    top_nodes = sorted_nodes[:10]  # Select the top 10 most influential nodes

    # Add number of outgoing edges as labels for top 10 nodes
    top_node_labels = {node[0]: f"{node[0]}\n[{node[1]}]" for node in top_nodes}
    nx.draw_networkx_labels(
        G, pos=node_positions, labels=top_node_labels,
        font_color='black', font_size=5, ax=ax
    )

    # node_sizes = [100 * degree + 80 for _, degree in top_nodes]
    node_colors = [hex_colors[int(key)] for key, _ in top_nodes]
    nx.draw_networkx_nodes(
        G, pos=node_positions, nodelist=[node[0] for node in top_nodes],
        node_color='white', edgecolors=node_colors, node_size=200, ax=ax
    )
    edge_list = G.out_edges([node[0] for node in top_nodes])
    edge_colors = [hex_colors[int(key)] for key, _ in edge_list]

    nx.draw_networkx_edges(
        G, pos=node_positions, edgelist=G.out_edges([node[0] for node in top_nodes]),
        edge_color=edge_colors, arrows=True, arrowstyle='->', width=1, ax=ax
    )
    ax.set_title("Strongest Dependencies")
    directory = 'Visualisations/' + config['modelVis']['default']+ '/horizon_' + config['horizonVis']['default'] + '/' + 'geographicVis/'
    filename = 'geoVis_split_' + split + '.png'
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath)

def plot_map_simple(adj_matrix, config):
    G = create_graph(adj_matrix, config)
    node_positions = nx.get_node_attributes(G, 'pos')

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # Extracting the min and max latitude and longitude for setting map boundaries
    min_lon = min(pos[0] for pos in node_positions.values())
    max_lon = max(pos[0] for pos in node_positions.values())
    min_lat = min(pos[1] for pos in node_positions.values())
    max_lat = max(pos[1] for pos in node_positions.values())
    width = max_lon - min_lon
    height = max_lat - min_lat

    m = Basemap(
        llcrnrlon=min_lon - 0.1 * width, llcrnrlat=min_lat - 0.1 * height,
        urcrnrlon=max_lon + 0.1 * width, urcrnrlat=max_lat + 0.1 * height,
        resolution='i', ax=ax
    )
    m.drawcoastlines(linewidth=0.5)
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='white', lake_color='lightblue')

    # Plot nodes on the map
    for node, (lon, lat) in node_positions.items():
        x, y = m(lon, lat)  # Convert lon, lat to x, y coordinates on the map
        m.plot(x, y, 'o', color='blue', markersize=6)  # 'o' for circle marker

    ax.set_title("Stations in South Africa")
    directory = 'Visualisations/' + config['modelVis']['default']+ '/horizon_' + config['horizonVis']['default'] + '/' + 'simpleMap/'
    filename =  '.png'
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath)
    
    # plt.show()

# Then call this function with necessary arguments:
# plot_map_simple(adj_matrix, config)


def plot_heatmap(adj_matrix, config, split):
    fig_heatmap, ax_heatmap = plt.subplots()
    sns.heatmap(adj_matrix, cmap='YlGnBu', ax=ax_heatmap)
    ax_heatmap.set_title("Adjacency Matrix Heatmap")
    directory = 'Visualisations/' + config['modelVis']['default']+ '/horizon_' + config['horizonVis']['default'] + '/' + 'heatmap/'
    filename = 'heatmap_split_' + split + '.png'
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    fig_heatmap.savefig(filepath)

def plot(config):
    number_of_splits = int(config['splitVis']['default'])
    for split in range(number_of_splits+1):
        split=str(split)
        matrix_path = "Results/" + config['modelVis']['default'] + "/" + config['horizonVis']['default'] + " Hour Forecast/Matrices/adjacency_matrix_" + split + ".csv"
        print("Matrix path: " + matrix_path)
        df = pd.read_csv(matrix_path, index_col=0)
        adj_matrix = df.values
        plot_map(adj_matrix, config , split)
        plot_map_simple(adj_matrix, config)
        plot_heatmap(adj_matrix, config, split)