import os
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from skimage.measure import label, find_contours, points_in_poly
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import shapely
from os import listdir
from os.path import isfile, join

cityname = ['chicago', 'austin', 'kitsap1', 'kitsap2', 'vienna'] 
h = 0
coord_scale = 1.0
template_width = 40
template_height = 4   # the accuracy of anchor matching rely on the density of template, for chicago, 20-by-2 is not enough


a = 1

def dist(anchor, target):
    dist_x = np.abs(anchor[:, 0] - target[0])
    dist_y = np.abs(anchor[:, 1] - target[1])
    dist = np.multiply(dist_x, dist_x) + np.multiply(dist_y, dist_y)
    return dist

def get_anchor_idx(dist, seq):
    if np.argmin(dist) not in seq:
        return np.argmin(dist)
    else:
        dist[np.argmin(dist)] = np.finfo(dist.dtype).max
        return get_anchor_idx(dist, seq)


def norm_geometry_to_graph(geometries, template_width, template_height):
    bldgnum = len(geometries)
    bounds = []
    size = []
    pos = []

    for i in range(bldgnum):
        bounds.append(geometries[i].bounds)
    
    bounds = np.array(bounds, dtype = np.double) # (minx, miny, maxx, maxy)
    minx = np.amin(bounds, axis = 0)[0]
    miny = np.amin(bounds, axis = 0)[1]
    maxx = np.amax(bounds, axis = 0)[2]
    maxy = np.amax(bounds, axis = 0)[3]

    lenx = maxx - minx
    leny = maxy - miny
        
    bounds[:, 0] = ( bounds[:, 0] - minx ) * 2.0 * coord_scale / lenx - coord_scale
    bounds[:, 2] = ( bounds[:, 2] - minx ) * 2.0 * coord_scale / lenx - coord_scale

    bounds[:, 1] = ( bounds[:, 1] - miny ) * 2.0 * coord_scale / leny - coord_scale
    bounds[:, 3] = ( bounds[:, 3] - miny ) * 2.0 * coord_scale / leny - coord_scale

    mx = np.mean( (bounds[:, 0], bounds[:, 2]) , axis = 0)
    my = np.mean( (bounds[:, 1], bounds[:, 3]) , axis = 0)
    
    size = np.stack((bounds[:, 2] - bounds[:, 0], bounds[:, 3] - bounds[:, 1]), axis = 1)
    size = size / 2.0
    pos = np.stack( (mx, my), axis = 1 )

    if lenx <= leny:   # swap x-y if x-length is shorter than y-length
        size[:, [0, 1]] = size[:, [1, 0]]
        pos[:, [0, 1]] = pos[:, [1, 0]]

    unit_w = np.double(2 * coord_scale) / np.double(template_width)
    unit_h = np.double(2 * coord_scale) / np.double(template_height)

    w_anchor = np.arange(-coord_scale + unit_w / 2.0, coord_scale + 1e-6, unit_w)
    h_anchor = np.arange(-coord_scale + unit_h / 2.0, coord_scale + 1e-6, unit_h)
    
    nearest_pos_seq = []

    anchorw = np.tile(w_anchor, len(h_anchor))
    anchorh = np.repeat(h_anchor, len(w_anchor))

    anchor = np.stack( (anchorw, anchorh), axis = 1)


    pos_sort = np.lexsort((pos[:,1],pos[:,0])) # The last column is the primary sort key.
    pos_sorted = pos[pos_sort]
    size_sorted = size[pos_sort]

    for i in range(bldgnum):
        dist_to_anchori = dist(anchor, pos_sorted[i, :])
        anchor_idx = get_anchor_idx(dist_to_anchori, nearest_pos_seq)
        nearest_pos_seq.append(anchor_idx)
    print(nearest_pos_seq)

    for i in range(bldgnum):
        print(anchor[nearest_pos_seq[i]], pos_sorted[i])

        
    # print(anchor)
    # print(nearest_pos_seq)

    return None    
    # pos_out = np.zeros((template_width * template_height, 2))
    # size_out = np.zeros_like(pos_out)
    # exist_out = np.zeros(template_width * template_height)
    
    # for i in range(bldgnum):
    #     idx = nearest_pos_seq[i]
    #     pos_out[idx] = pos[i, :]
    #     size_out[idx] = size[i, :]
    #     exist[idx] = 1

    
    # max_node = template_width * template_height
    # g = nx.grid_2d_graph(template_height, template_width)
    # G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')
    # for i in range(template_height):
    #     for j in range(template_width):
    #         idx = i * template_width + j
    #         G.nodes[idx]['posx'] = pos[idx, 0]
    #         G.nodes[idx]['posy'] = pos[idx, 1]
    #         G.nodes[idx]['exist'] = exist[idx]
    #         G.nodes[idx]['merge'] = 0
    #         G.nodes[idx]['size_x'] = size_out[idx, 0]
    #         G.nodes[idx]['size_y'] = size_out[idx, 1]

    # return G



if __name__ == "__main__":

    fp = 'D:\\OSM_dataset\\'+ cityname [h] +'_raw_set_all'
    if not os.path.exists(fp):
        os.mkdir(fp)

    dat_fp = 'D:\\OSM_dataset\\'+ cityname [h] +'_processed'
    if not os.path.exists(dat_fp):
        os.mkdir(dat_fp)

    save_fp = os.path.join(dat_fp, 'processed')
    if not os.path.exists(save_fp):
        os.mkdir(save_fp)

    vis_fp = os.path.join(dat_fp, 'visual')
    if not os.path.exists(vis_fp):
        os.mkdir(vis_fp)


    contourfiles = [f for f in listdir(fp) if isfile(join(fp, f)) and '.png' not in f  and  'bldg' not in f]
    bldgfiles = [f for f in listdir(fp) if isfile(join(fp, f)) and '.png' not in f  and  'road' not in f]

    assert len(bldgfiles) == len(contourfiles), "contour and bldg file total count is not the same."

    raw_num = len(contourfiles)
    # bldgnum_list = []
    # candid_idx_list = []
    c_idx = 0
    idx_map = {}

    for i in range(raw_num):
        # ct_f_idx = contourfiles[i][contourfiles[i].rfind('_')+1:]
        bg_f_idx = bldgfiles[i][bldgfiles[i].rfind('_')+1:]

        # assert ct_f_idx == bg_f_idx, "contour and bldg file index is not corresponded."

        # with (open(os.path.join(fp, contourfiles[i]), "rb")) as openfile:
        #     ct = pickle.load(openfile)   # Its x-y bounding box (ct.bounds) is a (minx, miny, maxx, maxy) tuple.  
  
        with (open(os.path.join(fp, bldgfiles[i]), "rb")) as openfile:
            bldg = pickle.load(openfile)
            bldgnum = len(bldg)
            # bldgnum_list.append(bldgnum)
            if bldgnum > 5 and bldgnum <= 50:
                # candid_idx_list.append(bg_f_idx)
                g = norm_geometry_to_graph(bldg, 40, 4)
                # nx.write_gpickle(g, os.path.join(save_fp, str(c_idx) + ".gpickle"), 4)
                # visual_block_graph(g, visual_path, str(c_idx), draw_edge = True, draw_nonexist = False)
                idx_map[c_idx] = bg_f_idx
                c_idx += 1

        if c_idx > 1:
            break
    print(idx_map)

    # bldgnum_arry = np.array(bldgnum_list, dtype = np.int16)
    # print(np.mean(bldgnum_arry), np.min(bldgnum_arry), np.max(bldgnum_arry), np.std(bldgnum_arry))

