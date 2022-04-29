import os
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
from skimage.measure import label, find_contours, points_in_poly
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import shapely
from os import listdir
from os.path import isfile, join
from test_visual import visual_block_graph
import json
import shapely.geometry as sg
import shapely.affinity as sa
from utils import included_angle, get_Block_azumith, get_RoadAccess_EdgeType, get_BldgRela_EdgeType, Ecu_dis, generate_RoadAccess_EdgeType
import shutil
import matplotlib.pyplot as plt
from Bldg_fit_func import fit_bldg_features
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def _azimuth(point1, point2):
    """azimuth between 2 points (interval 0 - 180)"""
    import numpy as np

    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180

def _dist(a, b):
    """distance between points"""
    import math

    return math.hypot(b[0] - a[0], b[1] - a[1])

def get_azimuth(mrr):
    """azimuth of minimum_rotated_rectangle"""
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])

    return az


def get_aspect_ratio(mrr):
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        aspect = np.double(axis1) / np.double(axis2)
    else:
        aspect = np.double(axis2) / np.double(axis1)

    return aspect


def get_size(mrr):
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        return axis2, axis1
    else:
        return axis1, axis2   # length, width




def get_bldggroup_parameters(bldg):
    multi_poly = MultiPolygon(bldg)
    bbx = multi_poly.minimum_rotated_rectangle
    azimuth = get_azimuth(bbx)
    return azimuth, bbx 



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



def norm_block_to_horizonal(bldg, azimuth, bbx):
    blk_offset_x = np.double(bbx.centroid.x)
    blk_offset_y = np.double(bbx.centroid.y)

    for i in range(len(bldg)):
        curr = sa.translate(bldg[i], -blk_offset_x, -blk_offset_y)
        bldg[i] = sa.rotate(curr, azimuth - 90, origin = (0.0, 0.0))

    return bldg




def norm_geometry_to_array(geometries):
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
    pos = np.stack( (mx, my), axis = 1 )

    if lenx <= leny:   # swap x-y if x-length is shorter than y-length
        size[:, [0, 1]] = size[:, [1, 0]]
        pos[:, [0, 1]] = pos[:, [1, 0]]


    pos_sort = np.lexsort((pos[:,1],pos[:,0])) # The last column is the primary sort key.
    pos_sorted = pos[pos_sort]
    size_sorted = size[pos_sort]
    
    return pos_sorted, size_sorted




def geoarray_to_graph(pos_sorted, size_sorted, blk_aspect_ratio, template_width, template_height):

    unit_w = np.double(2 * coord_scale) / np.double(template_width)
    unit_h = np.double(2 * coord_scale) / np.double(template_height)

    w_anchor = np.arange(-coord_scale + unit_w / 2.0, coord_scale + 1e-6, unit_w)
    h_anchor = np.arange(-coord_scale + unit_h / 2.0, coord_scale + 1e-6, unit_h)
    
    anchorw = np.tile(w_anchor, len(h_anchor))
    anchorh = np.repeat(h_anchor, len(w_anchor))
    anchor = np.stack( (anchorw, anchorh), axis = 1)


    bldgnum = pos_sorted.shape[0]
    nearest_pos_seq = []
    for i in range(bldgnum):
        dist_to_anchori = dist(anchor, pos_sorted[i, :])
        anchor_idx = get_anchor_idx(dist_to_anchori, nearest_pos_seq)
        nearest_pos_seq.append(anchor_idx)

    # for i in range(bldgnum):  # check the matching quality between nodes and anchor
    #     print(anchor[nearest_pos_seq[i]], pos_sorted[i])

    pos_out = np.zeros((template_width * template_height, 2))
    size_out = np.zeros_like(pos_out)
    exist_out = np.zeros(template_width * template_height)

    for i in range(bldgnum):
        idx = nearest_pos_seq[i]
        pos_out[idx] = pos_sorted[i, :]
        size_out[idx] = size_sorted[i, :]
        exist_out[idx] = 1
    
    max_node = template_width * template_height
    g = nx.grid_2d_graph(template_height, template_width)
    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')
    G.graph['aspect_ratio'] = blk_aspect_ratio
    for i in range(template_height):
        for j in range(template_width):
            idx = i * template_width + j
            G.nodes[idx]['posx'] = pos_out[idx, 0]
            G.nodes[idx]['posy'] = pos_out[idx, 1]
            G.nodes[idx]['exist'] = exist_out[idx]
            G.nodes[idx]['merge'] = 0
            G.nodes[idx]['size_x'] = size_out[idx, 0]
            G.nodes[idx]['size_y'] = size_out[idx, 1]

    return G


def read_filter_polygon(openfile):
    bldg = pickle.load(openfile)
    out = []
    for i in bldg:
        if i.geom_type == 'Polygon' and i.area > 4.0:
            out.append(i)
    return out



def remove_mutual_overlap(pos, size, intersect):
    int_bbx = intersect.bounds
    x_d = int_bbx[2] - int_bbx[0]
    y_d = int_bbx[3] - int_bbx[1]
    cx_d = intersect.centroid.x
    cy_d = intersect.centroid.y

    s_x = size[0]
    s_y = size[1]

    if np.double(x_d) / np.double(s_x) >= np.double(y_d) / np.double(s_y):
        if pos[1] >= cy_d:
            pos[1] = pos[1] + y_d / 2.0
        else:
            pos[1] = pos[1] - y_d / 2.0
        size[1] = size[1] - y_d
    else:
        if pos[0] >= cx_d:
            pos[0] = pos[0] + x_d / 2.0
        else:
            pos[0] = pos[0] - x_d / 2.0
        size[0] = size[0] - x_d

    return pos, size



def modify_geometry_overlap(bldg, iou_threshold = 0.5):
    bldgnum = len(bldg)
    rm_list = []
    pos = []
    size = []

    for i in range(bldgnum):
        pos.append([(bldg[i].bounds[0] + bldg[i].bounds[2]) / 2.0, (bldg[i].bounds[1] + bldg[i].bounds[3]) / 2.0])
        size.append([bldg[i].bounds[2] - bldg[i].bounds[0], bldg[i].bounds[3] - bldg[i].bounds[1] ])
    pos = np.array(pos)
    size = np.array(size)


    for i in range(bldgnum):
        for j in range(i+1, bldgnum):
            is_mod = False
            p1 = bldg[i]
            p2 = bldg[j]
            if p1.contains(p2):
                rm_list.append(i)
                continue
            if p2.contains(p1):
                rm_list.append(j)
                continue
            if p1.intersects(p2):
                intersect = p1.intersection(p2)
                int_area = intersect.area
                iou1 = int_area / p1.area
                iou2 = int_area / p2.area

                if iou1 > iou_threshold:
                    rm_list.append(i)
                    continue
                else:
                    pos[i,:], size[i,:] = remove_mutual_overlap(pos[i,:], size[i,:], intersect)
                    is_mod = True

                if iou2 > iou_threshold:
                    rm_list.append(j)
                    continue
                elif not is_mod:
                    pos[i,:], size[i,:] = remove_mutual_overlap(pos[i,:], size[i,:], intersect)
                
    pos = np.delete(pos, rm_list, axis=0)
    size = np.delete(size, rm_list, axis=0)

    bldg_list = []
    for i in range(pos.shape[0]):
        bldg_list.append(box(pos[i,0] - size[i,0] / 2.0, pos[i,1] - size[i,1] / 2.0, pos[i,0] + size[i,0] / 2.0, pos[i,1] + size[i,1] / 2.0))
    
    return bldg_list




def geometry_envelope(geometries):
    for i in range(len(geometries)):
        geometries[i] = geometries[i].envelope
    return geometries



def get_final_aspect_ratio(bldg_list):
    multi_poly = MultiPolygon(bldg_list)
    bbx = multi_poly.minimum_rotated_rectangle
    aspect_ratio = get_aspect_ratio(bbx)
    return aspect_ratio



def geometry_augment(bldg, cat_len = 3, flip_len = 3, rd_len = 1):   # concatenate will add "cat_len" base situtaion, and flip add "3*cat_len" situtations, random sampling added "rd_len*3*cat_len' situation
    bldg_list = [bldg]
 
    #######################  partial concatenate  ##############################
    if cat_len > 0:
        multi_poly = MultiPolygon(bldg)
        xmin = multi_poly.bounds[0]
        xmax = multi_poly.bounds[2]
        width = xmax - xmin
        int_len = cat_len + 2
        interval = np.linspace(xmin, xmax, int_len)  # (int_len-2) intervals, only on x-axis

        for i in range(1, int_len-1):
            cur_int = interval[i]
            cur_bldg = []
            for i in range(len(bldg)):
                if bldg[i].centroid.x < cur_int:
                    cur_bldg.append(sa.translate(bldg[i], width, 0))
                else:
                    cur_bldg.append(bldg[i])
            bldg_list.append(cur_bldg)
    #################################################################
    ############ Flip  ##############################################
    if flip_len > 0:
        mat_flip1 = [1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0]
        mat_flip2 = [-1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        for i in range(len(bldg_list)):
            num = len(bldg_list[i])
            cur_a = []
            cur_b = []
            cur_c = []

            if flip_len > 0:
                for j in range(num):
                    cur_a.append(sa.affine_transform(bldg_list[i][j], mat_flip1))
                bldg_list.append(cur_a)

            if flip_len > 1:
                for j in range(num):
                    cur_b.append(sa.affine_transform(bldg_list[i][j], mat_flip2))
                bldg_list.append(cur_b)

            if flip_len > 2:
                for j in range(num):
                    cur_c.append(sa.affine_transform(cur_a[j], mat_flip2))
                bldg_list.append(cur_c)      
    ##########################################################################
    ############ random sample  ##############################################
    if rd_len > 0:
        for i in range(len(bldg_list)):
            for k in range(rd_len):
                curr = []
                for j in range(len(bldg_list[i])):
                    cur_bldg = bldg_list[i][j]
                    b_w = cur_bldg.bounds[2] - cur_bldg.bounds[0]
                    b_h = cur_bldg.bounds[3] - cur_bldg.bounds[1]

                    mat_rd = [( 0.95 + np.random.random_sample() * 0.16), 0, 0, 
                    0, ( 0.95 + np.random.random_sample() * 0.16), 0, 
                    0, 0, 1, 
                    b_w * (-0.1 + np.random.random_sample() * 0.2), b_h * (-0.1 + np.random.random_sample() * 0.2), 0]
                    
                    curr.append(sa.affine_transform(bldg_list[i][j], mat_rd))
                bldg_list.append(curr)
    return bldg_list



def filter_little_intersected_bldglist(bldg, block):
    inside = []
    for bi in range(len(bldg)):
        if block.intersects(bldg[bi]):
            portion = np.double(block.intersection(bldg[bi]).area) / np.double(bldg[bi].area)
            if portion >= 0.5:
                inside.append(bldg[bi])
    return inside


def save_visual_block_bldg(fp, bldg_list, block, c_idx):
    plt.plot(*block.exterior.xy)            
    for kk in range(len(bldg)):
        plt.plot(*bldg[kk].exterior.xy)
    plt.savefig(os.path.join(fp, str(c_idx) + '.png'))
    plt.clf()





def get_bldg_features(bldg):

    resolution = 0.3
    x, y = bldg.exterior.xy
    minx = np.amin(x)
    miny = np.amin(y)
    maxx = np.amax(x)
    maxy = np.amax(y)
    
    x = x - minx
    y = y - miny
    
    width = np.double(maxx - minx) / resolution # Width of pixel in 0.3m resolution
    height = np.double(maxy - miny) / resolution # Height of pixel in 0.3m resolution
    
    dpi = 400
    w_inch = width / np.double(dpi)
    h_inch = height / np.double(dpi)
    
    fig = plt.figure(figsize=(w_inch, h_inch), dpi=dpi)
    plt.fill(x, y)
    
    ax = fig.gca()
    ax.axis('off')
    fig.tight_layout(pad=0)
    
    # To remove the huge white borders
    ax.margins(0)
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    as_rgba = np.frombuffer(img_as_string, dtype='uint8').reshape((height, width, 4))
    
    img = as_rgba[:, :, :3]
    plt.clf()
    plt.close(fig)

    curr_shape, iou, curr_height, curr_width, theta = fit_bldg_features(img)

    print(curr_shape, iou, curr_height, curr_width, theta)














if __name__ == "__main__":


    cityname = ['chicago', 'washington', 'nyc'] 
    h = 2
    coord_scale = 1.0
    template_width = 25
    template_height = 2   # the accuracy of anchor matching rely on the density of template, for chicago, 20-by-2 is not enough
    N = template_width * template_height
    # plt.subplots(figsize=(20, 4))
    check_visual = 200

    rd_len = 0   ### total number = sample * (rd_len + 1) * (flip_len + 1) * (cat_len + 1)
    cat_len = 0
    flip_len = 0

    min_bldg = 20   # >
    max_bldg = 50   # <=

    dat_fp = '/opt/data/liuhe95/osm_dataset/'+ cityname[h] + '0.5_graph_dataset'
    if not os.path.exists(dat_fp):
        os.mkdir(dat_fp)

    dat_fp = os.path.join(dat_fp, 'filter')
    if not os.path.exists(dat_fp):
        os.mkdir(dat_fp)

    save_fp = os.path.join(dat_fp, 'processed')
    if not os.path.exists(save_fp):
        os.mkdir(save_fp)

    rawvis_fp = os.path.join(dat_fp, 'raw_visual')
    if not os.path.exists(rawvis_fp):
        os.mkdir(rawvis_fp)

    transvis_fp = os.path.join(dat_fp, 'raw_visual_after_trans')
    if not os.path.exists(transvis_fp):
        os.mkdir(transvis_fp)


    fltvis_fp = os.path.join(dat_fp, 'filter_visual')
    if not os.path.exists(fltvis_fp):
        os.mkdir(fltvis_fp)

    bldgnum_list = []
    block_size = []
    block_asp_rto = []
    c_idx = 0
    bldgfiles = []
    visfiles = []
    roadfiles = []
    reserved_idx = []
    
    filtered_bldg_file = []
    filtered_road_file = []
    filtered_vis_file = []


    for jj in range(12):
        fp = '/opt/data/liuhe95/osm_dataset/'+cityname[h]+'_0.5_full_set/'+cityname[h]+'_0.5_'+str(jj)+'_full'
        vis_path = os.path.join(fp, 'visual')
        bldg_path = os.path.join(fp, 'bldg')
        road_path = os.path.join(fp, 'road')


        curr = [os.path.join(bldg_path, f) for f in listdir(bldg_path) if isfile(join(bldg_path, f))]
        bldgfiles = bldgfiles + curr

        curr1  = [os.path.join(vis_path, f) for f in listdir(vis_path) if isfile(join(vis_path, f))]
        visfiles = visfiles + curr1

        curr2  = [os.path.join(road_path, f) for f in listdir(road_path) if isfile(join(road_path, f))]
        roadfiles = roadfiles + curr2



    raw_num = len(bldgfiles)
    raw_rnum = len(roadfiles)
    assert raw_num == raw_rnum, "contour and bldg file index is not corresponded {}, {}.".format(raw_num, raw_rnum)


    for i in range(raw_num):

        with (open(bldgfiles[i], "rb")) as openfile:
            bldg = read_filter_polygon(openfile)
            block = pickle.load(open(roadfiles[i], "rb"))
            longside, _ = get_size(block.minimum_rotated_rectangle)

            ############    fitler block is too large > 300m
            if longside > 300.0:
                print(str(i), ', block too large > 300m.')
                continue

            ############    fitler bldg that is not more than halved covered by the block contour 
            bldg = filter_little_intersected_bldglist(bldg, block)
            if len(bldg) == 0:
                print(str(i), ', no building after filtering.')
                continue

            ############    fitler block contour too irregular             
            portion = np.double(block.area) / np.double(block.minimum_rotated_rectangle.area)
            multi_poly = MultiPolygon(bldg)

            if portion < 0.5:
                print(str(i), ', block: ', portion)
                # save_visual_block_bldg(fltvis_fp, bldg, block, str(i) + '_blk' + str(portion))
                continue

            elif portion < 0.8:
                bldg_convex_portion = np.double(multi_poly.convex_hull.area) / np.double(multi_poly.minimum_rotated_rectangle.area)
                ############    if building groups in irregular block contour looks good, keep them
                if bldg_convex_portion < 0.8:
                    print(str(i), ", building: ", bldg_convex_portion, ', block: ', portion)
                    # save_visual_block_bldg(fltvis_fp, bldg, block, str(i) + '_blk' + str(portion) + '_bu' + str(bldg_convex_portion))
                    continue
            

            bldg_convex_tt = np.double(multi_poly.convex_hull.area) / np.double(block.area)
            if bldg_convex_tt < 0.2:
                print(str(i), ", total building convex: ", bldg_convex_tt)
                # save_visual_block_bldg(fltvis_fp, bldg, block, str(i) + '_tconv' + str(bldg_convex_tt))
                continue

            if len(bldg) < 6:
                tt_bldg = 0
                intsct_bldg = 0

                for bbi in range(len(bldg)):
                    tt_bldg += np.double(bldg[bbi].area)
                    intsct_bldg += np.double(block.intersection(bldg[bbi]).area)

                tt_bldg = np.double(tt_bldg) / np.double(block.area)
                intsct_bldg = np.double(intsct_bldg) / np.double(block.area)
                if tt_bldg < 0.3:
                    print(str(i), ", total building: ", tt_bldg)
                    # save_visual_block_bldg(fltvis_fp, bldg, block, str(i) + '_tt' + str(tt_bldg))
                    continue
                elif intsct_bldg < 0.2:
                    print(str(i), ", intersect building: ", intsct_bldg)
                    # save_visual_block_bldg(fltvis_fp, bldg, block, str(i) + '_intt' + str(intsct_bldg))
                    continue

            ############   building number after filtering
            reserved_idx.append(i)
            bldgnum = len(bldg)
            bldgnum_list.append(bldgnum)
            # save_visual_block_bldg(rawvis_fp, bldg, block, c_idx)
            c_idx += 1
            print("{}, good sample, {}, portion: {}".format(i, c_idx, portion))
            block_size.append(get_size(block.minimum_rotated_rectangle))
            block_asp_rto.append(get_aspect_ratio(block.minimum_rotated_rectangle))

            filtered_bldg_file.append(bldgfiles[i])
            filtered_road_file.append(roadfiles[i])
            filtered_vis_file.append(visfiles[i])


    # with open(os.path.join(dat_fp, 'raw_to_selected_index_map.txt'), 'w') as file_map:
    #     file_map.write(json.dumps(idx_map))
    

    bldgnum_list = np.array(bldgnum_list)
    print(np.mean(bldgnum_list), np.std(bldgnum_list), bldgnum_list.shape)
    bins = np.arange(min(bldgnum_list), max(bldgnum_list))
    plt.hist(bldgnum_list, bins=bins, alpha=0.5) 
    # plt.xlim(0, 100)
    plt.savefig(os.path.join(dat_fp, 'num.png'))
    plt.clf()      


    block_size = np.array(block_size)
    long_s = block_size[:, 0]
    short_s = block_size[:, 1]
    print(np.mean(long_s), np.std(long_s), long_s.shape)
    print(np.mean(short_s), np.std(short_s), short_s.shape)

    bins = np.arange(min(long_s), max(long_s))
    plt.hist(long_s, bins=bins, alpha=0.5) 
    # plt.xlim(0, 300)
    plt.savefig(os.path.join(dat_fp, 'long_side.png'))
    plt.clf()  


    bins = np.arange(min(short_s), max(short_s))
    plt.hist(short_s, bins=bins, alpha=0.5) 
    # plt.xlim(0, 150)
    plt.savefig(os.path.join(dat_fp, 'short_side.png'))
    plt.clf()  

    
    block_asp_rto = np.array(block_asp_rto)
    bins = np.arange(min(block_asp_rto), max(block_asp_rto), 0.01)
    plt.hist(block_asp_rto, bins=bins, alpha=0.5) 
    plt.savefig(os.path.join(dat_fp, 'aspect_ratio.png'))
    plt.clf()  


    reserved_idx = np.array(reserved_idx)
    with open(os.path.join(dat_fp,'reserved_idx.data'), 'wb') as f:
        pickle.dump(reserved_idx, f)

    with open(os.path.join(dat_fp,'bldgnum.data'), 'wb') as f:
        pickle.dump(bldgnum_list, f)

    with open(os.path.join(dat_fp,'block_size.data'), 'wb') as f:
        pickle.dump(block_size, f)

    with open(os.path.join(dat_fp,'block_asp_rto.data'), 'wb') as f:
        pickle.dump(block_asp_rto, f)


    with open(os.path.join(dat_fp,'bldg_file.data'), 'wb') as f:
        pickle.dump(filtered_bldg_file, f)

    with open(os.path.join(dat_fp,'road_file.data'), 'wb') as f:
        pickle.dump(filtered_road_file, f)

    with open(os.path.join(dat_fp,'vis_file.data'), 'wb') as f:
        pickle.dump(filtered_vis_file, f)