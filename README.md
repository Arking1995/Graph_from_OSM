# Graph_from_OSM


Process pipeline of graph block dataset from OSM: 
 
	Get_Geometry_from_OSM.py  (input lat and lon of bounding box, paralleled by multiprocess.)  
	combine_dataset_from_tiles.py   (combine dataset from different processes.)  
	filter_dataset.py        (filter clean dataset from raw dataset, save all file index and statstics under './filter' at the dataset level folder.)  
	generate_graph_from_filtered_dataset.py    (dense grid or sparse grid from filtered dataset.)  
 
 
