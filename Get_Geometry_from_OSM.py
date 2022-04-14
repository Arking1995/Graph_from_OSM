import osmnx as ox
import os
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from skimage.measure import label, find_contours, points_in_poly
from skimage.color import label2rgb
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import pickle
import multiprocessing

import warnings


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





def find_nearest_bigger(array, value):
    if value <= array[0]:
        return 0
    if value >= array[array.size-1]:
        return array.size-1

    for idx in range(array.size):
        if value <= array[idx]:
            return idx
        

def find_nearest_smaller(array, value):
    if value <= array[0]:
        return 0

    if value >= array[array.size-1]:
        return array.size-1

    for idx in range(1, array.size):
        if value >= array[idx-1] and value <= array[idx]:
            return idx-1






def process_osm(input):
    bbx, idx = input
    colsize = 30
    rowsize = 30

    fp = os.path.join('D:\\OSM_dataset\\chicago_7', str(idx))
    if not os.path.exists(fp):
        os.mkdir(fp)

    network_type='all'

    tags = {"building": True}

    print('process ',idx , ' start.')

    try:
        G = ox.graph_from_bbox(bbx[0], bbx[1], bbx[2], bbx[3], network_type=network_type, simplify=True, truncate_by_edge = True)
    except:
        print('process ', idx, ', downloading graph from bbx failed.')
        return False, idx   

    G_projected = ox.project_graph(G)
    H = k_core(G, 2)

    if H.number_of_nodes() < 5 or H.number_of_edges() < 5:
        print('process ', idx, ', not enough nodes in the graph.')
        return False, idx

    try:
        fig1, ax1 = ox.plot_graph(H, node_size=0, bgcolor='#ffffff',  edge_color='#000000', edge_linewidth=1, figsize = (colsize, rowsize), save=False, show=False, close=True)
    except:
        print(idx, ', not enough nodes in the graph.')
        return False, idx

    try: 
        all_footprints = ox.geometries_from_bbox(bbx[0], bbx[1], bbx[2], bbx[3], tags)
    except:
        print('process ', idx, ', downloading geometries from bbx failed.')
        return False, idx   

    print('process ', idx, ', downloading finished.')
    ft_size = all_footprints.shape[0]
    all_footprints['index_col'] = range(ft_size) #add a column for indexing



    img = plot2img(fig1)
    label_image = label(img > 128) # find connected region and label them
    image_label_overlay = label2rgb(label_image[:,:,0], image=img[:,:,0])


    blk_num = len(np.unique(label_image.ravel()))
    print(idx, ', Total contour: ', blk_num - 2)
    print(idx, ', Total bldg polygon: ', ft_size)

    un_processed = set(range(ft_size))


    x = H.nodes.data('x')
    y = H.nodes.data('y')
    xy = np.array([(x[node], y[node]) for node in H.nodes])
    # eps = (xy.max(axis=0) - xy.min(axis=0)).mean() / 100
    eps = 5e-4



    bounds_list = []
    poly_list = []
    for i in range(ft_size):
        cur = all_footprints.loc[all_footprints['index_col']==i,'geometry'][0]
        if cur.geom_type == 'Polygon':
            bounds_list.append(cur.bounds)
            poly_list.append(cur)

    bounds_list = np.array(bounds_list, dtype = np.double)
    spatial_order = np.lexsort((bounds_list[:,1],bounds_list[:,0]))  # minx, miny order from low to high

    poly_minx_sorted = bounds_list[spatial_order][:, 0]
    ccount = 0

    fig2, ax2 = plt.subplots()
    for ii in np.unique(label_image.ravel()):
        if ii == 0 or ii == 1:
            continue

        inside_ft_idx = []
        proj_poly = []

        # print('Process ', idx, ', contour processing: ', ii)
        mask = (label_image[:,:,0] == ii)
        # cv2.imwrite(os.path.join(fp,'image_'+str(ii)+'.png'),mask.astype(np.uint8) * 255.0)

        contours = find_contours(mask.astype(np.float64), 0.5)
        # Select the largest contiguous contour
        contour = sorted(contours, key=lambda x: len(x))[-1]

        if len(contour) < 10:
            continue


        # display the image and plot the contour;
        # this allows us to transform the contour coordinates back to the original data cordinates
        ax2.imshow(mask, interpolation='nearest', cmap='gray')
        ax2.autoscale(enable=False)
        ax2.step(contour.T[1], contour.T[0], linewidth=1, c='r')
        plt.clf()


        # # first column indexes rows in images, second column indexes columns;
        # # therefor we need to swap contour array to get xy values
        contour = np.fliplr(contour)
        pixel_to_data = ax2.transData + ax2.transAxes.inverted() + ax1.transAxes + ax1.transData.inverted()
        transformed_contour = pixel_to_data.transform(contour)

        curr_blk = Polygon(transformed_contour)

        b_minx = curr_blk.bounds[0]
        b_maxx = curr_blk.bounds[2]

        start_idx = find_nearest_bigger(poly_minx_sorted, b_minx)
        end_idx = find_nearest_smaller(poly_minx_sorted, b_maxx)
        
        if start_idx == ft_size or end_idx == 0:
            continue

        for j in range(start_idx, end_idx):
            if spatial_order[j] in un_processed:
                if curr_blk.intersects(poly_list[spatial_order[j]]):
                    inside_ft_idx.append(spatial_order[j])
                    un_processed.remove(spatial_order[j])


        if len(inside_ft_idx)>0:
            print('process ', idx, ', blk ', ccount, ' contains: ', len(inside_ft_idx))
            with open(os.path.join(fp,'blk_road_'+str(ccount)), "wb") as poly_file:
                obb_blk = ox.projection.project_geometry(curr_blk)[0]
                pickle.dump(obb_blk, poly_file, pickle.DEFAULT_PROTOCOL)

            for k in range(len(inside_ft_idx)):
                proj_poly.append(ox.projection.project_geometry(poly_list[inside_ft_idx[k]])[0])

            with open(os.path.join(fp,'blk_bldg_'+str(ccount)), "wb") as poly_file:
                pickle.dump(proj_poly, poly_file, pickle.DEFAULT_PROTOCOL)

            for geom in proj_poly:
                if geom.geom_type == 'Polygon':
                    plt.plot(*geom.exterior.xy)            
            plt.savefig(os.path.join(fp,'blk_'+str(ccount)+'.png'))
            plt.clf()

            ccount += 1

    print('process ', idx, ', function output finished ')
    plt.close('all') 
    return True, idx











if __name__ == '__main__':

    fp = 'D:\\OSM_dataset\\chicago_7'
    if not os.path.exists(fp):
        os.mkdir(fp)

    # cityname = ['chicago', 'austin', 'kitsap1', 'kitsap2', 'vienna']    #'chicago', 'austin', 'kitsap', 'vienna'

    # bbx = [ [41.9058124, 41.8241064, -87.6209515, -87.7285386],
    #         [30.2985978, 30.2165930, -97.6955337, -97.7899633],
    #         [47.5244795, 47.4576584, -122.5890458, -122.7082366],
    #         [47.5783890, 47.5516808, -122.6275684, -122.6879689],
    #         [48.2386057, 48.1580527, 16.4240489, 16.3034633] ]

    # bbx = [ [41.8241064, 41.7424004, -87.6209515, -87.7285386]]  ####'chicago_new'
    # bbx = [ [41.9058124, 41.8241064, -87.6209515, -87.7285386]] ####'chicago_origin'
    # bbx = [ [41.9058124, 41.8241064, -87.7285386, -87.83612569]]  ####'chicago_new2'



    bbx = [ [41.9875184, 41.8241064, -87.7285386, -87.83612569],  #### '3'
            [41.9875184, 41.8241064, -87.6209515, -87.7285386],   #### '4'
            [42.0692244, 41.9875184, -87.7285386, -87.83612569],  #### '5'
            [42.0692244, 41.9875184, -87.6209515, -87.7285386],    #### '6'
            [41.8241064, 41.7424004, -87.7285386, -87.83612569],   #### '7'
            [41.7424004, 41.6606944, -87.7285386, -87.83612569],   #### '8'
            [41.7424004, 41.6606944,  -87.6209515, -87.7285386]    #### '9'
            ]



    til_col = 8
    til_row = 8  
    sub_bbx = []
    unfinished = set(range(til_col * til_row))
    finished = set()

    h = 4
    til_col_arr = np.arange(bbx[h][2], bbx[h][3] + np.double(bbx[h][3] - bbx[h][2]) / np.double(til_col), np.double(bbx[h][3] - bbx[h][2]) / np.double(til_col))
    til_row_arr = np.arange(bbx[h][0], bbx[h][1] + np.double(bbx[h][1] - bbx[h][0]) / np.double(til_row), np.double(bbx[h][1] - bbx[h][0]) / np.double(til_row))

    for row in range(til_row):
        for col in range(til_col):
            sub_bbx.append([til_row_arr[row], til_row_arr[row+1], til_col_arr[col], til_col_arr[col+1]])


    input_list = []
    for idx, b in enumerate(sub_bbx):
        input_list.append([b, idx])

    count_procs = 0
    processer_num = 8
    with multiprocessing.Pool(processer_num) as pool:
        for i, iswork in enumerate(pool.imap_unordered(process_osm, input_list), 1):
            isf, idx = iswork
            unfinished.remove(idx)
            finished.add(idx)
            count_procs += 1 
            print('Finished total {} process. Total {} processes.'.format(count_procs, len(input_list)))
            print('Left processes: ', unfinished, ' unfinished.')
            print('Finished processes: ', finished)            
            if isf:
                print('the process {} finished successfully.'.format(idx))
            else:
                print('the process {} failed.'.format(idx))

    print('Finish all processes.')





















    # curr_blk = gpd.GeoDataFrame()
    # curr_blk.loc[0, 'geometry'] = Polygon(transformed_contour)
    # curr_blk.gdf_name = 'curr_blk'
    # curr_blk.crs = {'init' :'epsg:4326'}
    # curr_blk = curr_blk.unary_union.convex_hull

    # gdf_temp = ox.projection.project_gdf(curr_blk1, to_latlong=True)

    # footprints = ox.geometries_from_polygon(curr_blk1 , tags)
    # if footprints.size > 0:
    #     fig, ax = ox.plot_footprints(footprints, filepath=os.path.join(fp,'blk_'+str(ii)+'.png'), dpi=800, save=True, show=False, close=True)



# transformed_contour_path = Path(transformed_contour, closed=True)
    # print(transformed_contour_path.shape)
    # patch = PathPatch(transformed_contour_path, facecolor='red')
    # ax1.add_patch(patch)


    # x = G.nodes.data('x')
    # y = G.nodes.data('y')
    # xy = np.array([(x[node], y[node]) for node in G.nodes])
    # eps = (xy.max(axis=0) - xy.min(axis=0)).mean() / 10.0
    #
    # is_inside = transformed_contour_path.contains_points(xy, radius=-eps)
    # nodes_inside_block = [node for node, flag in zip(G.nodes, is_inside) if flag]
    #
    # boundary = []
    # for node in nodes_inside_block:
    #     boundary.append([ G.nodes[node]['x'], G.nodes[node]['y']])
    #






    # node_size = [50 if node in nodes_inside_block else 0 for node in G.nodes]
    # node_color = ['r' if node in nodes_inside_block else 'k' for node in G.nodes]
    # fig3, ax3 = ox.plot_graph(G, node_color=node_color, node_size=node_size)


