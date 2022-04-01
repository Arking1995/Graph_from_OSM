import osmnx as ox
import os
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from skimage.measure import label, find_contours, points_in_poly
from skimage.color import label2rgb
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from Block_Graph import BlockGraph
import pickle
import geopandas as gpd
from shapely.strtree import STRtree

def k_core(G, k):
    H = nx.Graph(G, as_view=True)
    H.remove_edges_from(nx.selfloop_edges(H))
    core_nodes = nx.k_core(H, k)
    H = H.subgraph(core_nodes)
    return G.subgraph(core_nodes)


def plot2img(fig):
    # remove margins
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    # convert to image
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    as_rgba = np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))
    return as_rgba[:,:,:3]



cityname = ['chicago', 'austin', 'kitsap1', 'kitsap2', 'vienna']    #'chicago', 'austin', 'kitsap', 'vienna'

bbx = [ [41.9058124, 41.8241064, -87.6209515, -87.7285386],
        [30.2985978, 30.2165930, -97.6955337, -97.7899633],
        [47.5244795, 47.4576584, -122.5890458, -122.7082366],
        [47.5783890, 47.5516808, -122.6275684, -122.6879689],
        [48.2386057, 48.1580527, 16.4240489, 16.3034633] ]

network_type='all'
# The index for the name of the city
# h = 2

for k in range(1, 5):
    h = k
    colsize = 30
    rowsize = 30

    if h == 2 or h == 3:
        til_col = 5
        til_row = 5
    else:
        til_col = 10
        til_row = 10       

    til_col_arr = np.arange(bbx[h][2], bbx[h][3] + np.double(bbx[h][3] - bbx[h][2]) / np.double(til_col), np.double(bbx[h][3] - bbx[h][2]) / np.double(til_col))
    til_row_arr = np.arange(bbx[h][0], bbx[h][1] + np.double(bbx[h][1] - bbx[h][0]) / np.double(til_row), np.double(bbx[h][1] - bbx[h][0]) / np.double(til_row))



    fp = 'D:\\OSM_dataset\\'+ cityname[h] + '_raw_para_' + network_type
    if not os.path.exists(fp):
        os.mkdir(fp)

    global_output_idx = 0

    for row in range(til_row):
        for col in range(til_col):
        
            # fp1 = os.path.join(fp, str(row) + '_' + str(col))
            # if not os.path.exists(fp1):
            #     os.mkdir(fp1)
            fp1 = fp

            tags = {"building": True}

            print('start: row ' , row, 'col: ', col)
            # print(til_row_arr[row], til_row_arr[row+1], til_col_arr[col], til_col_arr[col+1])

            G = ox.graph_from_bbox(til_row_arr[row], til_row_arr[row+1], til_col_arr[col], til_col_arr[col+1], network_type=network_type)

            # G_projected = ox.project_graph(G)
            H = k_core(G, 2)
            if H.number_of_nodes() == 0 or H.number_of_edges() == 0:
                continue
            fig1, ax1 = ox.plot_graph(H, node_size=0, bgcolor='#ffffff',  edge_color='#000000', edge_linewidth=1, figsize = (colsize, rowsize),save=True, show=False, close=True)

            all_footprints = ox.geometries_from_bbox(bbx[h][0], bbx[h][1], bbx[h][2], bbx[h][3], tags)

            # all_footprints = ox.project_gdf(all_footprints)
            ft_size = all_footprints.shape[0]
            all_footprints['index_col'] = range(ft_size) #add a column for indexing
            processed = set(range(ft_size))

            # USE STRtree
            # all_geom = all_footprints.loc[:,'geometry'].values
            # tree = STRtree(all_geom)


            img = plot2img(fig1)
            label_image = label(img > 128) # find connected region and label them
            image_label_overlay = label2rgb(label_image[:,:,0], image=img[:,:,0])
            cv2.imwrite(os.path.join(fp1,'image_label_overlay'+ str(row) + '_' + str(col) +'.png'),image_label_overlay * 255.0)

            print('Total contour: ', len(np.unique(label_image.ravel())) - 2)


            x = H.nodes.data('x')
            y = H.nodes.data('y')
            xy = np.array([(x[node], y[node]) for node in H.nodes])
            # eps = (xy.max(axis=0) - xy.min(axis=0)).mean() / 100
            eps = 5e-4

            for ii in np.unique(label_image.ravel()):
            # ii = np.argsort(np.bincount(label_image.ravel()))[-5]
                if ii == 0 or ii == 1:
                    continue

                print('idx: ', global_output_idx)
                mask = (label_image[:,:,0] == ii)
                # cv2.imwrite(os.path.join(fp1,'image_'+str(ii)+'.png'),mask.astype(np.uint8) * 255.0)

                contours = find_contours(mask.astype(np.float64), 0.5)
                # Select the largest contiguous contour
                contour = sorted(contours, key=lambda x: len(x))[-1]

                if len(contour) < 10:
                    continue

                # display the image and plot the contour;
                # this allows us to transform the contour coordinates back to the original data cordinates
                fig2, ax2 = plt.subplots()
                ax2.imshow(mask, interpolation='nearest', cmap='gray')
                ax2.autoscale(enable=False)
                ax2.step(contour.T[1], contour.T[0], linewidth=1, c='r')
                plt.close(fig2)


                # # first column indexes rows in images, second column indexes columns;
                # # therefor we need to swap contour array to get xy values
                contour = np.fliplr(contour)
                pixel_to_data = ax2.transData + ax2.transAxes.inverted() + ax1.transAxes + ax1.transData.inverted()
                transformed_contour = pixel_to_data.transform(contour)
                curr_blk = Polygon(transformed_contour).convex_hull


                # proj_blk = ox.projection.project_geometry(curr_blk)
                # transformed_contour_path = Path(transformed_contour, closed=True)
                # is_inside = transformed_contour_path.contains_points(xy, radius=-eps)
                # nodes_inside_block = [node for node, flag in zip(H.nodes, is_inside) if flag]




                # # USE STRtree
                # within_bldg = [o for o in tree.query(curr_blk)]
                # if len(within_bldg)>0:
                #     obb_blk = curr_blk.minimum_rotated_rectangle
                #     # blk_footprint = all_footprints.iloc[inside_ft_idx]
                #     # bldg_polygon = list(all_footprints.iloc[inside_ft_idx]['geometry'])

                #     with open(os.path.join(fp1,'blk_road_'+str(global_output_idx)), "wb") as poly_file:
                #         obb_blk = ox.projection.project_geometry(obb_blk)[0]
                #         # print(obb_blk)
                #         pickle.dump(obb_blk, poly_file, pickle.DEFAULT_PROTOCOL)

                #     for j in range(len(within_bldg)):
                #         within_bldg[j] = ox.projection.project_geometry(within_bldg[j])[0]
                #         # print(within_bldg[j])

                #     with open(os.path.join(fp1,'blk_bldg_'+str(global_output_idx)), "wb") as poly_file:
                #         pickle.dump(within_bldg, poly_file, pickle.DEFAULT_PROTOCOL)

                #     d = {'id': range(len(within_bldg)), 'geometry': within_bldg}

                #     fig, ax = ox.plot_footprints(gpd.GeoDataFrame(d, crs="EPSG:3857"), filepath=os.path.join(fp1,'blk_'+str(global_output_idx)+'.png'), dpi=800, save=True, show=False, close=True)
                    
                #     global_output_idx += 1



                inside_ft_idx = []
                for i in processed.copy():
                    p1 = all_footprints.loc[all_footprints['index_col']==i,'geometry'][0]
                    if p1.intersects(curr_blk):
                        inside_ft_idx.append(i)
                        processed.remove(i)

                if len(inside_ft_idx)>0:
                    obb_blk = curr_blk.minimum_rotated_rectangle
                    blk_footprint = all_footprints.iloc[inside_ft_idx]
                    bldg_polygon = list(all_footprints.iloc[inside_ft_idx]['geometry'])

                    with open(os.path.join(fp1,'blk_road_'+str(global_output_idx)), "wb") as poly_file:
                        obb_blk = ox.projection.project_geometry(obb_blk)[0]
                        # print(obb_blk)
                        pickle.dump(obb_blk, poly_file, pickle.DEFAULT_PROTOCOL)

                    for j in range(len(bldg_polygon)):
                        bldg_polygon[j] = ox.projection.project_geometry(bldg_polygon[j])[0]
                        # print(bldg_polygon[j])

                    with open(os.path.join(fp1,'blk_bldg_'+str(global_output_idx)), "wb") as poly_file:
                        pickle.dump(bldg_polygon, poly_file, pickle.DEFAULT_PROTOCOL)

                    fig, ax = ox.plot_footprints(blk_footprint, filepath=os.path.join(fp1,'blk_'+str(global_output_idx)+'.png'), dpi=800, save=True, show=False, close=True)
                    
                    global_output_idx += 1
