import networkx as nx
from networkx import DiGraph
import os, re
import pickle
from shapely import geometry
import matplotlib.pyplot as plt
from Bldg_fit_func import fit_bldg_features
from Block_Graph import BlockGraph
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import shapely.geometry as sg
import shapely.affinity as sa
from utils import included_angle, get_Block_azumith, get_RoadAccess_EdgeType, get_BldgRela_EdgeType, Ecu_dis, generate_RoadAccess_EdgeType
from shapely.strtree import STRtree
import random
from matplotlib.patches import Rectangle


coord_cutoff = 1000.0
directP_dict = {(0,'north'),(1,'south'),(2,'east'),(3,'west')}
resolution = 0.3
road_access_threshold = 20.0
bldg_rela_threshold = 10.0
thres_mean_size = 5 # also defined in utils.py, they should be the same in case road direction cannot be identified by generate_RoadAccess_EdgeType()
np.random.seed(42)


plt.subplots(figsize=(20, 4))


def visual_grid_graph(G, filepath, filename):
    pos = {}
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    for ee in range(G.number_of_nodes()):
        pos[ee] = (G.nodes[ee]['posx'], G.nodes[ee]['posy'])

    nx.draw_networkx(G, pos=pos,
            node_color='lightgreen',
            with_labels=True,
            node_size=600)
    plt.savefig(os.path.join(filepath,filename))
    plt.clf()




def generate_graph_fts_from_exist_num(g, height, width, existed_num, coord_scale):
    max_node = height * width
    exist_idx = random.sample(range(40), existed_num)
    exist = np.zeros(max_node, dtype = np.int32)
    exist[exist_idx] = 1

    dy = np.double(2 * coord_scale) / np.double(height + 1)
    dx = np.double(2 * coord_scale) / np.double(width + 1)

    x_coord = np.arange(-coord_scale + dx, coord_scale - dx + 1e-6, dx)
    y_coord = np.arange(-coord_scale + dy, coord_scale - dy + 1e-6, dy)

    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')
    for i in range(height):
        for j in range(width):
            f_x = np.random.normal(0.0, dx / 5.0, 1)[0]
            f_y = np.random.normal(0.0, dy / 5.0, 1)[0]
            idx = i * width + j
            G.nodes[idx]['posy'] = y_coord[i] + f_y
            G.nodes[idx]['posx'] = x_coord[j] + f_x
            if exist[idx] == 0:
                G.nodes[idx]['exist'] = 0
                G.nodes[idx]['merge'] = 0
            else:
                G.nodes[idx]['exist'] = 1
                G.nodes[idx]['merge'] = 0
    # print(G.nodes(data=True))
    return G




def generate_graph_fts_from_existpattern(g, height, width, exist_pattern, coord_scale):
    max_node = height * width

    dy = np.double(2 * coord_scale) / np.double(height + 1)
    dx = np.double(2 * coord_scale) / np.double(width + 1)

    x_coord = np.arange(-coord_scale + dx, coord_scale - dx + 1e-6, dx)
    y_coord = np.arange(-coord_scale + dy, coord_scale - dy + 1e-6, dy)

    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')
    for i in range(height):
        for j in range(width):
            f_x = np.random.normal(0.0, dx / 5.0, 1)[0]
            f_y = np.random.normal(0.0, dy / 5.0, 1)[0]
            idx = i * width + j
            G.nodes[idx]['posy'] = y_coord[i] + f_y
            G.nodes[idx]['posx'] = x_coord[j] + f_x
            if exist_pattern[idx] == 0:
                G.nodes[idx]['exist'] = 0
                G.nodes[idx]['merge'] = 0
            else:
                G.nodes[idx]['exist'] = 1
                G.nodes[idx]['merge'] = 0
    # print(G.nodes(data='exist'))
    return G



def generate_exist_pattern(interval, length):
    exist = np.zeros(length, dtype = np.int32)
    is_exist = False
    for i in range(length):
        if i % interval == 0:
            is_exist = not is_exist
        if is_exist:
            exist[i] = 1
    return exist







def generate_bldgsize_seq(row, col):
    out = {}
    out[0] = generate_col_bldgsize_seq(col)
    if row == 1:
        return out

    for i in range(1, row):
        out[i] = generate_col_bldgsize_seq(col)
    return out


def generate_col_bldgsize_seq(leng):
    seq = np.random.choice(np.arange(1,4), leng, p=[0.7, 0.2, 0.1])
    summ = 0
    out_seq = []
    for i in range(leng):
        if summ >= leng:
            if summ > leng:
                out_seq[-1] = out_seq[-1] - (summ - leng)
            break
        out_seq.append(seq[i])
        summ += seq[i]
    return out_seq


def closest(lst, K):
     idx = (np.abs(lst - K)).argmin()
     return idx


def sparse_generate_pos_and_exist_from_bldgsize_seq(height, width, bldgsize, coord_scale, masked):
    max_node = height * width

    unit_w = np.double(2 * coord_scale) / np.double(width)
    unit_h = np.double(2 * coord_scale) / np.double(height)

    x_anchor = np.arange(-coord_scale + unit_w / 2.0, coord_scale + 1e-6, unit_w)
    y_anchor = np.arange(-coord_scale + unit_h / 2.0, coord_scale + 1e-6, unit_h)

    pos_x_seq = {}
    pos_y_seq = {}

    w_seq = {}
    h_seq = {}

    for i in range(len(bldgsize)): # row
        seq = bldgsize[i]
        x = []
        y = []
        s_w = []
        s_h = []
        cur_x = -coord_scale
        cur_y = -coord_scale + unit_h * i

        for j in range(len(seq)): # col
            w = seq[j] * unit_w * ( 0.9 + np.random.random_sample() * 0.2)
            h = unit_h * ( 0.9 + np.random.random_sample() * 0.2)
            s_w.append(w)
            s_h.append(h)        
            x.append(cur_x + w/2.0)
            y.append(cur_y + h/2.0)
            cur_x += w

        pos_x_seq[i] = np.array(x, dtype=np.float32)
        pos_y_seq[i] = np.array(y, dtype=np.float32)
        w_seq[i] = np.array(s_w, dtype=np.float32)
        h_seq[i] = np.array(s_h, dtype=np.float32)

    nearest_pos_seq = {}
    for i in range(len(pos_x_seq)):
        cur = []
        for j in range(len(pos_x_seq[i])):
            cur.append(closest(x_anchor, pos_x_seq[i][j])) # now only consider nearest in width direction
        nearest_pos_seq[i] = np.array(cur, dtype = np.int16)


    x_pos = np.zeros(height * width)
    y_pos = np.zeros(height * width)
    exist = np.zeros(height * width)
    w_out = np.zeros(height * width)
    h_out = np.zeros(height * width)
    
    # ##################### uncomment for anchor input for non-exist nodes, comment for defualt 0 input for non-exist node
    if not masked:
        for i in range(height):
            for j in range(width):
                x_pos[i * width + j] = x_anchor[j]
                y_pos[i * width + j] = y_anchor[i]
    # ##################### uncomment for anchor input for non-exist nodes, comment for defualt 0 input for non-exist node


    for i in range(len(nearest_pos_seq)):
        for j in range(len(nearest_pos_seq[i])):
            w_idx = nearest_pos_seq[i][j]
            x_pos[i * width + w_idx] = pos_x_seq[i][j]
            y_pos[i * width + w_idx] = pos_y_seq[i][j]
            h_out[i * width + w_idx] = h_seq[i][j]
            w_out[i * width + w_idx] = w_seq[i][j]
            exist[i * width + w_idx] = 1
    
    return [x_pos, y_pos, h_out, w_out, exist]
    

    

# def dense_generate_pos_and_exist_from_bldgsize_seq(height, width, bldgsize, coord_scale):




def sparse_generate_graph_from_ftsarray(height, width, x_pos, y_pos, h_out, w_out, exist):
    max_node = height * width
    g = nx.grid_2d_graph(height, width)
    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            G.nodes[idx]['posy'] = y_pos[idx]
            G.nodes[idx]['posx'] = x_pos[idx]
            G.nodes[idx]['exist'] = exist[idx]
            G.nodes[idx]['merge'] = 0
            G.nodes[idx]['size_x'] = w_out[idx]
            G.nodes[idx]['size_y'] = h_out[idx]
    return G





if __name__ == "__main__":

    save_fp = 'D:\\OSM_dataset\\synthetic_size_test'
    if not os.path.exists(save_fp):
        os.mkdir(save_fp)

    save_input = os.path.join(save_fp, 'processed')
    if not os.path.exists(save_input):
        os.mkdir(save_input)

    save_visual = os.path.join(save_fp, 'visual')
    if not os.path.exists(save_visual):
        os.mkdir(save_visual)

    col_sz = 20
    row_sz = 2
    normalize_scale = 1

    blk_wd = 10
    blk_ht = 10

    max_node = 40

    idx = 0
    fn_ct = 0

    sample = 20
    checkvisual = 2000

    ismasked = False

    # pattern_dict = []
    # pattern_dict.append(generate_exist_pattern(1, 40))
    # pattern_dict.append(generate_exist_pattern(2, 40))
    # pattern_dict.append(generate_exist_pattern(4, 40))
    # pattern_dict.append(generate_exist_pattern(8, 40))
    # pattern_dict.append(generate_exist_pattern(10, 40))
    # pattern_dict.append(generate_exist_pattern(20, 40))

    # for j in range(sample):
    #     for i in range(len(pattern_dict)):
    #         g_add = generate_graph_fts_from_existpattern(nx.grid_2d_graph(2, 20), 2, 20, pattern_dict[i], 1)
    #         visual_grid_graph(g_add, save_visual, str(fn_ct) + '.png')
    #         nx.write_gpickle(g_add, os.path.join(save_input, str(fn_ct) + ".gpickle"), 4)
    #         fn_ct += 1

 
    for i in range(sample):
        bldgsize_dict = generate_bldgsize_seq(2, 20)
        x_pos, y_pos, h_out, w_out, exist = sparse_generate_pos_and_exist_from_bldgsize_seq(row_sz, col_sz, bldgsize_dict, normalize_scale, ismasked)

        g_add = sparse_generate_graph_from_ftsarray(row_sz, col_sz, x_pos, y_pos, h_out, w_out, exist)
        nx.write_gpickle(g_add, os.path.join(save_input, str(fn_ct) + ".gpickle"), 4)

        if i % checkvisual == 0:
            visual_grid_graph(g_add, save_visual, str(fn_ct) + '.png')
            print('Computing ', i, '.....')    
        
        fn_ct += 1
    print('Finish')


