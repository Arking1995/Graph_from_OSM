import os
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import shapely
from os import listdir
from os.path import isfile, join
import json
import shutil
import matplotlib.pyplot as plt


def read_filter_polygon(openfile):
    bldg = pickle.load(openfile)
    out = []
    for i in bldg:
        if i.geom_type == 'Polygon' and i.area > 4.0:
            out.append(i)
    return out



path = '/opt/data/liuhe95/osm_dataset/chicago_0.5_6'
if not os.path.exists(path):
    os.mkdir(path)

out_path = '/opt/data/liuhe95/osm_dataset/chicago_0.5_6_full'
if not os.path.exists(out_path):
    os.mkdir(out_path)

vis_path = os.path.join(out_path, 'visual')
if not os.path.exists(vis_path):
    os.mkdir(vis_path)

bldg_path = os.path.join(out_path, 'bldg')
if not os.path.exists(bldg_path):
    os.mkdir(bldg_path)

road_path = os.path.join(out_path, 'road')
if not os.path.exists(road_path):
    os.mkdir(road_path)


ccount = 0

for i in range(144):
    fp = os.path.join(path, str(i))

    road_file_list = [f for f in listdir(fp) if isfile(join(fp, f)) and '.png' not in f  and  'bldg' not in f]
    bldg_file_list = [f for f in listdir(fp) if isfile(join(fp, f)) and '.png' not in f  and  'road' not in f]
    vis_file_list = [f for f in listdir(fp) if isfile(join(fp, f)) and '.png' in f]

    assert len(bldg_file_list) == len(road_file_list), "contour and bldg file total count is not the same at {}.".format(i)


    raw_num = len(bldg_file_list)
    bldg_list = []


    for ii in range(raw_num):
        ct_f_idx = road_file_list[ii][road_file_list[ii].rfind('_')+1:]
        bg_f_idx = bldg_file_list[ii][bldg_file_list[ii].rfind('_')+1:]
        assert ct_f_idx == bg_f_idx, "contour and bldg file index is not corresponded at {}.".format(i)

        # with (open(os.path.join(fp, bldg_file_list[i]), "rb")) as openfile:
        #     bldg = read_filter_polygon(openfile)
        #     bldg_list.append(bldg)
        

    # dup_id = set()
    # for i in range(len(bldg_list)):
    #     for j in range( (max(0, i - 5)), min(i+5, len(bldg_list)) ):
    #         if i != j:
    #             for ii in range(len(bldg_list[i])):
    #                 for jj in range(len(bldg_list[j])):
    #                     if bldg_list[i][ii].almost_equals(bldg_list[j][jj]):
    #                         print(road_file_list[i], road_file_list[j])
    #                         if len(bldg_list[i]) >= len(bldg_list[j]):
    #                             dup_id.add(j)
    #                         else:
    #                             dup_id.add(i)
    #                         break


    for j in range(len(bldg_file_list)):
        rst = os.path.join(fp, road_file_list[j])
        dst = os.path.join(road_path, str(ccount))
        shutil.copyfile(rst, dst)
        
        rst = os.path.join(fp, bldg_file_list[j])
        dst = os.path.join(bldg_path, str(ccount))
        shutil.copyfile(rst, dst)

        rst = os.path.join(fp, vis_file_list[j])
        dst = os.path.join(vis_path, str(ccount) + '.png')
        shutil.copyfile(rst, dst)

        ccount += 1
