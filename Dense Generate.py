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


coord_cutoff = 1000.0
directP_dict = {(0,'north'),(1,'south'),(2,'east'),(3,'west')}
resolution = 0.3
road_access_threshold = 20.0
bldg_rela_threshold = 10.0
thres_mean_size = 5 # also defined in utils.py, they should be the same in case road direction cannot be identified by generate_RoadAccess_EdgeType()

if __name__ == "__main__":

    fp = 'D:\\Sat_road\\chicago_set_raw'
    save_fp = 'D:\\Sat_road\\chicago_dense_dataset'

    # fp = 'D:\\Sat_road\\result_newtest\\testblk_cv_chicago_drive'
    # save_fp = 'D:\\Sat_road\\chicago_trivial_dataset'

    idx_list = []
    rx = re.compile(r"blk_bldg_*")
    for root, dirs, files in os.walk(fp):
        for file in files:
            res = re.match(rx, file)
            if res:
                idx_list.append(file[9:])


    if not os.path.exists(fp):
        os.mkdir(fp)

    if not os.path.exists(save_fp):
        os.mkdir(save_fp)


    for ii in range(1): #len(idx_list)

        with open(os.path.join(fp,'blk_bldg_'+ idx_list[ii]), "rb") as poly_file:
            loaded_polygon = pickle.load(poly_file)

        with open(os.path.join(fp,'blk_road_'+ idx_list[ii]), "rb") as poly_file:
            loaded_road = pickle.load(poly_file)

        # print(len(loaded_polygon))
        poly_vol = len(loaded_polygon)

        blk_offset_x = np.float64(loaded_road.centroid.x)
        blk_offset_y = np.float64(loaded_road.centroid.y)


        ################################ Generate block-level graph attributes
        offset_block = sa.translate(loaded_road, -blk_offset_x, -blk_offset_y)
        blk_bounds = offset_block.bounds
        blk_azumith = get_Block_azumith(offset_block)
        horiz_offset_block = sa.rotate(offset_block, blk_azumith, origin = (0.0, 0.0), use_radians=True)
        blk_area = horiz_offset_block.area

        if blk_bounds[2] - blk_bounds[0] < thres_mean_size or blk_bounds[3] - blk_bounds[1] < thres_mean_size:
            continue

        G = BlockGraph(blockID= ii, offsetx = blk_offset_x , offsety = blk_offset_y, area = blk_area, azimuth = blk_azumith) # initialize the block graph

        ################################ Generate road nodes
        x_rd, y_rd = horiz_offset_block.exterior.xy

        sum_angle = 0.0
        line_list = []

        # block has been rotated to horizontal, save included angle and length of road
        line_vol = 4
        for i in range(line_vol):
            curr_road = sg.LineString([(x_rd[i], y_rd[i]), (x_rd[i+1], y_rd[i+1])])
            line_list.append(curr_road)
            rd_length = curr_road.length
            if i < line_vol - 1:
                G.add_obj_node(i, posx=curr_road.centroid.x, posy=curr_road.centroid.y, road_length = rd_length)
            else:
                G.add_obj_node(i, posx=curr_road.centroid.x, posy=curr_road.centroid.y, road_length = rd_length)

        road_access_dict = generate_RoadAccess_EdgeType(line_list)
        # print(road_access_dict)

        line_vol = len(line_list)
        ######################### Connect all road nodes
        for i in range(line_vol):
            if i < line_vol - 1:
                eu_dist = Ecu_dis(G.nodes[i]['posx'], G.nodes[i]['posy'], G.nodes[i+1]['posx'], G.nodes[i+1]['posy'])
                sum_angle += tmp_angle
                edge_azumith, edge_type = get_BldgRela_EdgeType(line_list[i].centroid,
                                                                    line_list[i+1].centroid)
                G.add_obj_edge(i, i+1, edge_dist=eu_dist, included_angle=tmp_angle, azimuth =edge_azumith, edge_type = edge_type)
            else:
                eu_dist = Ecu_dis(G.nodes[i]['posx'], G.nodes[i]['posy'], G.nodes[0]['posx'], G.nodes[0]['posy'])
                tmp_angle = 2 * np.pi - sum_angle
                edge_azumith, edge_type = get_BldgRela_EdgeType(line_list[i].centroid,
                                                                    line_list[0].centroid)
                G.add_obj_edge(i, 0, edge_dist=eu_dist, included_angle=tmp_angle, azimuth =edge_azumith, edge_type = edge_type)


        ################################ Generate building nodes
        horiz_offset_bldg = []
        for i in range(poly_vol):
            if loaded_polygon[i].geom_type == 'Polygon':
                tmp_offset = sa.translate(loaded_polygon[i], -blk_offset_x, -blk_offset_y)
                tmp_rot = sa.rotate(tmp_offset, blk_azumith, origin=(0.0, 0.0), use_radians=True)
                horiz_offset_bldg.append(tmp_rot)
                x_rd, y_rd = tmp_rot.exterior.xy

        poly_vol_valid = len(horiz_offset_bldg)



        Rtree = STRtree(horiz_offset_bldg)
        for i in range(poly_vol_valid):
            # print('processing ', str(i), 'th building.')

            curr_poly = horiz_offset_bldg[i]

            curr_area = curr_poly.area
            curr_posx = curr_poly.centroid.x
            curr_posy = curr_poly.centroid.y

            nearest_poly = Rtree.nearest(curr_poly)
            nearest_posx = nearest_poly.centroid.x
            nearest_posy = nearest_poly.centroid.y



            G.add_obj_node(i + 4, posx=curr_posx, posy=curr_posy, bldg_area=curr_area)





        ####### link edges of all road access bldgs
        for j in range(poly_vol_valid):
            tmp_bldg = horiz_offset_bldg[j]
            distance_list = []
            for i in range(line_vol):
                tmp_dist = line_list[i].distance(tmp_bldg)
                distance_list.append(tmp_dist)
                if (tmp_dist < road_access_threshold):
                    tmp_direct = list(road_access_dict.keys())[list(road_access_dict.values()).index(i)]
                    rd_azumith, rd_edgetype = get_RoadAccess_EdgeType(tmp_direct)
                    G.add_obj_edge(i, j + 4, edge_dist=tmp_dist, azimuth=rd_azumith, edge_type=rd_edgetype)

            dist_arr = np.array(distance_list)
            minid = np.argmin(dist_arr)

            if (dist_arr[minid] >= road_access_threshold): # initially any bldg node should have at least 1 road access
                tmp_direct = list(road_access_dict.keys())[list(road_access_dict.values()).index(i)]
                rd_azumith, rd_edgetype = get_RoadAccess_EdgeType(tmp_direct)
                G.add_obj_edge(minid, j + 4, edge_dist=dist_arr[minid], edge_type=rd_edgetype)
                # a point distance to a line, should always be vertical, orientation will be used by edges between buildings
                # north, south, west, east. edge_type defined by global directP_dict{}

        for i in range(poly_vol_valid):
            for j in np.arange(i+1, poly_vol_valid):
                tmp_dist = horiz_offset_bldg[i].centroid.distance(horiz_offset_bldg[j].centroid)
                if tmp_dist < bldg_rela_threshold:
                    bldg_azumith, bldg_edgetype = get_BldgRela_EdgeType(horiz_offset_bldg[i].centroid, horiz_offset_bldg[j].centroid)
                    G.add_obj_edge(i + 4, j + 4, edge_dist=tmp_dist, azimuth =bldg_azumith, edge_type = bldg_edgetype)

        print(G.edges)
        G2 = G.to_undirected()
        print(G2.get_edge_data(0, 1))
        print(G2.get_edge_data(1, 0))
        nx.write_gpickle(G2, os.path.join(save_fp, idx_list[ii] +".gpickle"))





