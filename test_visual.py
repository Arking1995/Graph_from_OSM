import networkx as nx
import os, re
import pickle
from shapely import geometry
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

# plt.subplots(figsize=(20, 4))


def visual_block_graph(G, filepath, filename, draw_edge = False, draw_nonexist = False):
    pos = []
    size = []
    edge = []
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

    if not draw_nonexist:
        for i in range(G.number_of_nodes()):
            if G.nodes[i]['exist'] == 0: # or abs(G.nodes[i]['size_x']) < 1e-2 or abs(G.nodes[i]['size_y']) < 1e-2 s
                G.remove_node(i)

    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    for i in range(G.number_of_nodes()):
        pos.append([G.nodes[i]['posx'], G.nodes[i]['posy']])
        size.append([G.nodes[i]['size_x'], G.nodes[i]['size_y']])

    for e in G.edges:
        edge.append(e)

    pos = np.array(pos, dtype = np.double)    
    size = np.array(size, dtype = np.double)
    edge = np.array(edge, dtype = np.int16)

    plt.scatter(pos[:, 0], pos[:, 1], c = 'red', s=50)
    ax = plt.gca()
    for i in range(size.shape[0]):
        ax.add_patch(Rectangle((pos[i, 0] - size[i, 0] / 2.0, pos[i, 1] - size[i, 1] / 2.0), size[i, 0], size[i, 1], linewidth=2, edgecolor='r', facecolor='b', alpha=0.3)) 

    if draw_edge:
        for i in range(edge.shape[0]):
            l = mlines.Line2D([pos[edge[i, 0], 0], pos[edge[i, 1], 0]], [pos[edge[i, 0], 1], pos[edge[i, 1], 1]])
            ax.add_line(l)

    plt.savefig(os.path.join(filepath,filename + '.png'))
    plt.clf()



if __name__ == "__main__":
    # f_path = 'D:\\Graph Dataset Generation\\VAE Results\\test_synthetic_size_large'
    f_path = 'D:\\OSM_dataset\\synthetic_size_large'

    g_path = os.path.join(f_path, 'processed')
    # visual_path = os.path.join(f_path, 'visual')
    visual_path = os.path.join('D:\\OSM_dataset', 'visual')

    fn_ct = 0
    f_list = os.listdir(g_path)

    for i in f_list[:20]:
        g = nx.read_gpickle(os.path.join(g_path, i))  
        visual_block_graph(g, visual_path, i, draw_edge = True, draw_nonexist = False)

