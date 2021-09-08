# importing libraries
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from scipy.ndimage import distance_transform_edt
import pandas as pd
import itertools
import matplotlib
import networkx as nx
import folium
import statistics
from sklearn.preprocessing import LabelEncoder
import matplotlib.colors as colors
import matplotlib.cm as cmx
import sys
import os
import io
import PIL 

def block_to_shape(block_name):
  '''
  This proc takes in the LA downtown name and extracts Graphs from it
  The path and city are hardcoded but can easily be modified 
  to make it more flexible
  '''

  place = block_name+' , Los Angeles'
  try:
    G = ox.graph_from_place(place)
    nodes,edges = ox.graph_to_gdfs(G)
    ox.save_graph_geopackage(G, filepath="/content/sample_data/LA Shape Files/"+block_name+"/"+block_name+".shp")
    return nodes,edges
  except:
    print("Data not present for:-"+block_name)
    
    
    
def return_first_element(x):
  '''
  In the OSM metadata for highway certain elements have list
  This proc is used to return the first element of the list 
  '''
  
  if isinstance(x,list):
    return x[0]
  else:
    return x  

def SpeedLimitColumnFormatter(edges):
  '''
  This proc is used to format columns which will later be used in Speed to join with Speed Limit Master
  The DataFrame contains two added columns higway_single_element and max_speed
  highway_single_element converts a list classification of roads frpm list to string giving single element
  max_speed contains cleanes speed limits with mph removed
  '''
  #df_speed = pd.unique(edges[['highway', 'maxspeed']].values())
  highway_single_element = [return_first_element(x) for x in edges['highway']]
  max_speed=edges['maxspeed'].astype(str).str[0:3]
  max_speed.replace('nan',np.nan,inplace=True)
  max_speed= pd.to_numeric(max_speed, errors='coerce')
  edges['highway_single_element']=highway_single_element
  edges['max_speed']=max_speed


  return edges


def SpeedMasterFileGenerator(file_path,edges):
  '''
  This function generates the master file for speed Limit
  There are many speed limits for the same category hence we will take the median values
  The ouput of this function is a file which can then be used to impute missing values
  '''

  edges_speed_master=pd.DataFrame(edges[['max_speed','highway_single_element']])
  edges_speed_master.dropna(inplace=True)
  edges_speed_master.reset_index(drop=True, inplace=True)
  
  df = pd.DataFrame(edges_speed_master.groupby('highway_single_element')['max_speed'].median())
  df.to_csv(file_path+'SPEED_LIMIT_MASTER.CSV')
  
  #return pd.DataFrame(edges_speed_master.groupby('highway_single_element')['max_speed'].median())
  
  
  
def MaxSpeedRiskScoreCalculator(master_file_path,edge):
  '''
  This function takes in the path of the Speed Limit Master File and imputes missing 
  values in the Edge dataframe
  This column would then be used in the Risk Score Calculation
  The edges dataframe should have preprocessed max_speed highway_single_element Elements
  These are generated from the above functions 
  For na Medain of the median values are filled
  '''
  df_speed_master = pd.read_csv(master_file_path)
  df_merged = edge.merge(df_speed_master,how='left',on='highway_single_element')  
  df_merged.max_speed_x.fillna(df_merged.max_speed_y, inplace=True)
  df_merged.drop('max_speed_y', axis=1, inplace=True) 
  #df_merged.rename(columns={'max_speed_x': 'max_speed_risk_score'}, inplace=True) 
  df_merged['max_speed_risk_score'].fillna(df_merged['max_speed_risk_score'].median(),inplace=True)
  return df_merged


def export_to_shape(edges_extended,file_path,block_name):
    '''
    This function is used to export the geodataframe files to shape files 
    '''

    edges_extended_temp=edges_extended.copy()
    edges_extended_temp['max_speed_risk_score']=edges_extended_temp['max_speed_risk_score'].astype(int)
    edges_extended_temp[['geometry','length','max_speed_risk_score']].to_file("/content/sample_data/LA Shape Files/"+block_name+"/"+block_name+".shp",driver ='ESRI Shapefile')


def Line_to_list_lonlats(geom, summary, lon_index, lat_index):
  '''
  Function developed by Sanjana Tule
  '''
  #geom.bounds is a tuple consisting of (lower_lat, lower_lon, upper_lat, upper_lon)
  delta_x = geom.bounds[2]-geom.bounds[0]
  delta_y = geom.bounds[3]-geom.bounds[1]
  #print("delta_x",delta_x)
  #print("delta_y",delta_y)
  steps_num = int(np.ceil(max(delta_x/abs(lon_index[1]-lon_index[0]), delta_y/abs(lat_index[1]-lat_index[0]))))
  #print("steps_num",steps_num)
  coords_list = [geom.interpolate(i, normalized=True).coords[0] for i in np.linspace(0,1,steps_num)]
  return coords_list

def lonlat_to_xy_to_values(list_, summary, lon_index, lat_index, np_matrix, filter_size = 1):
  '''
  Function developed by Sanjana Tule
  '''
  list_x = [i[0] for i in list_]
  lon_min = lon_index.min()  
  list_x = [x if x >= lon_min else lon_min for x in list_x ]
  lon_max = lon_index.max()
  list_x = [x if x <= lon_max else lon_max for x in list_x ]
  

  list_y = [i[1] for i in list_]
  lat_min = lat_index.min() 
  list_y = [y if y >= lat_min else lat_min for y in list_y ]
  lat_max = lat_index.max()
  list_y = [y if y <= lat_max else lat_max for y in list_y ]


  x = [np.nonzero((lon>=lon_index))[0][-1] for lon in list_x] 
  y = [np.nonzero((lat>=lat_index))[0][0] for lat in list_y]

  point_distance = []
  for i,v in zip(x,y):
    #print("i:{} and v:{}".format(i,v))
    temp=[]
    for j in itertools.product(range(i-1,i+2),range(v-1,v+2)):
      if (j[1]<np_matrix.shape[0])&(j[0]<np_matrix.shape[1]) and j[0] >=0 and j[1] >=0:
        #print("j[0]:{} and j[1]:{}".format(j[0],j[1]))
        #print("matrix value",np_matrix[j[1],j[0]])
        temp.append(np_matrix[j[1],j[0]])
    #print("distance array",temp)
    #print("max",max(temp))
    point_distance.append(max(temp))
    #print("point_distance",point_distance)

  #arrays = [np.array([np_matrix[j[1],j[0]] if (j[0]<np_matrix.shape[1])&(j[1]<np_matrix.shape[1]) else 0 for j in itertools.product(range(i-filter_size,i+filter_size+1),range(v-filter_size,v+filter_size+1))]).min() for i,v in zip(x,y)]
  
  x_y = [[i,v] for i,v in zip(x,y)]
  # print("x_y",x_y)
  lon_lat = [[i,v] for i,v in zip(list_x,list_y)]
  # print("lon_lat",lon_lat)
  return point_distance, x_y, lon_lat

def extract_value_from_matrix(geom, summary, lat_index, lon_index, np_matrix, filter_size = 1):
  '''
  Function developed by Sanjana Tule
  '''
  
  #print("geom",geom)
  list_ = Line_to_list_lonlats(geom, summary, lon_index, lat_index)
  #print("list",list_)
  assert len(list_) > 0, 'list is empty'
  arrays, x_y, lon_lat = lonlat_to_xy_to_values(list_, summary, lon_index, lat_index, np_matrix, filter_size)
  return {'sampling_counts': len(arrays),
          'max': max(arrays),
          'min': min(arrays),
          'mean': sum(arrays)/len(arrays),
          'median': statistics.median(arrays),
          'x_y': x_y,
          'lon_lat': lon_lat}   


def CalcualteDistanceTransform(block_name):
    '''
    This code is developed by Snajana Tule
    Here we are collacting all the steps necessary to add DistanceRiskScore to the matrix
    '''
    
    place = block_name+' , Los Angeles'	
    buildings_gdf   = ox.geometries_from_place(place, {'building':True})
    
    geocode_gdf = ox.geocode_to_gdf(place)


    # get the latitude and longitude of the area
    lon_max, lon_min = geocode_gdf[['bbox_west']].values[0][0], geocode_gdf[['bbox_east']].values[0][0]
    lat_max, lat_min = geocode_gdf[['bbox_north']].values[0][0], geocode_gdf[['bbox_south']].values[0][0]
    
    path = '/content/sample_data/segmented.png'
    
    buildings_box   = ox.geometries_from_bbox(lat_min,lat_max,lon_min,lon_max, {'building':True})
    
    ## Codes Commeted from Here 
    fig, ax = ox.plot_footprints(buildings_box, figsize=(8, 8),color='black',bgcolor='#FFFFFF',dpi=100)
    buildings_gdf_box   = ox.geometries_from_bbox(lat_min,lat_max,lon_min,lon_max, {'building':True})
    fig, ax = ox.plot_footprints(buildings_box, figsize=(8, 8),color='black',bgcolor='#FFFFFF',dpi=100)
    
    
    
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img=PIL.Image.open(buf)
    
    
    arr=np.asarray(img)
    img=arr[:,:,0]
    
    idx=[]
    for i in range(np.shape(img)[0]-1):
      if len(np.unique(img[i])) == 1:
        idx.append(i)
    
    img=np.delete(img,idx, axis=0)
    
    idx=[]
    for i in range(np.shape(img)[1]-1):
      if len(np.unique(img[:,i])) == 1:
        idx.append(i)
    
    img=np.delete(img,idx, axis=1)    
    
    #########CHANGED BY HUSSAIN#################
    
    ret, bw_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    distance = distance_transform_edt(bw_img)
    
    distance.shape
    distance = np.round(distance)

    return distance
    ## Codes Commeted from Here 


    
    return distance

def EdgesExtended(block_name,edges,distance):
    '''
    This procedure was developed by Sanajana Tule
    It takes in block name,distance,edges as arguments 
    It returns the Extended Edges dataframe
    '''
    
    
    
    place = block_name+' , Los Angeles'    
    # get the area ( will need later for plotting )
    area_gdf = ox.geocode_to_gdf(place)    
    # get the latitude and longitude of the area
    lon_max, lon_min = area_gdf[['bbox_west']].values[0][0], area_gdf[['bbox_east']].values[0][0]
    lat_max, lat_min = area_gdf[['bbox_north']].values[0][0], area_gdf[['bbox_south']].values[0][0] 
    

    # merge the distance transform and the osmnx graph
    summary = pd.DataFrame({'Name': ['x','y'],'max': [lon_max, lat_max],'min': [lon_min, lat_min]}).set_index('Name')
    summary['np_matrix pixels'] = [distance.shape[1], distance.shape[0]] 
    #display(summary)
    
    # generate latitude and longitude evenly spaced
    lat_index = np.flip(np.linspace(summary['min'].y,summary['max'].y,summary['np_matrix pixels'].y))
    lon_index = np.linspace(summary['min'].x,summary['max'].x,summary['np_matrix pixels'].x) 


    # get information for every edge in the graph   
    _ = edges.geometry.apply(lambda x: extract_value_from_matrix(x, summary, lat_index, lon_index, distance, filter_size = 1))

    # merge the additional information with the existing edges dataframe
    edges_extended = edges.merge(pd.DataFrame({ 'sampling_counts' : _.apply(lambda x: x['sampling_counts']),
                                            'max': _.apply(lambda x: x['max']),
                                            'min': _.apply(lambda x: x['min']),
                                            'mean': _.apply(lambda x: x['mean']),
                                            'median': _.apply(lambda x: x['median']),
                                            'x_y': _.apply(lambda x: x['x_y']),
                                            'lon_lat': _.apply(lambda x: x['lon_lat'])}), left_index=True, right_index=True)

    edges_extended['mean_scaled'] = edges_extended['mean'].apply(lambda x: np.exp(np.interp(x, (edges_extended['mean'].min(), edges_extended['mean'].max()), (0, 5))))
    edges_extended['median_scaled'] = edges_extended['median'].apply(lambda x: np.exp(np.interp(x, (edges_extended['median'].min(), edges_extended['median'].max()), (0, 5))))
    edges_extended['mean_scaled_inversed'] = edges_extended['mean_scaled'].apply(lambda x: abs(x - edges_extended['mean_scaled'].max()))
    edges_extended['median_scaled_inversed'] = edges_extended['median_scaled'].apply(lambda x: abs(x - edges_extended['median_scaled'].max()))
    edges_extended['distance_risk_score'] = edges_extended['max'].apply(lambda x: abs(x - edges_extended['max'].max()))
    
    return edges_extended       


def SaveToPicke(file_path,file_name,edges,nodes):

  '''
  This proc is used to save the edges geodataframe to a picklefile
  '''

  new_graph = ox.graph_from_gdfs(nodes,edges)
  nx.write_gpickle(new_graph, file_path+file_name)

def BuildingDensityCalculator(block_name,edges):
    
    '''
    This proc adds building densities to the dataframe.
    The master files for the are preperared using QGIS tool
    '''
    
    
    #G=ox.graph_from_place('NorthRidge ,Los Angeles')
    gdf = ox.geocoder.geocode_to_gdf(block_name+", Los Angeles")
    #nodes, edges = ox.graph_to_gdfs(G)
    box = gdf[["bbox_north", "bbox_south", "bbox_east", "bbox_west"]]
    
    df=pd.read_csv('/content/drive/MyDrive/Silicon Valley Earthquake Challenge/Building Density Masters/'+block_name+'.csv')
    
    latitudes = np.linspace( box ["bbox_north"].iloc[0],  box["bbox_south"].iloc[0], 20)
    longitudes = np.linspace( box["bbox_west"].iloc[0],  box["bbox_east"].iloc[0], 20)
    
    edges['Apartment Density']=0
    edges['Commercial Density']=0
    edges['Housing Density']=0
    edges['Industrial Density']=0
    edges['Retail Density']=0


    for i in range(edges.shape[0]):
        a=edges["geometry"].iloc[i].bounds
        df['New']=abs(df['Left']-a[0])+abs(df['Top']-a[1])+abs(df['Right']-a[2])+abs(df['Bottom']-a[3])
        df.sort_values('New',inplace=True)
        edges['Apartment Density'].iloc[i]=df['Apartment Density'].iloc[0]
        edges['Commercial Density'].iloc[i]=df['Commercial Density'].iloc[0]
        edges['Housing Density'].iloc[i]=df['Housing Density'].iloc[0]
        edges['Industrial Density'].iloc[i]=df['Industrial Density'].iloc[0]
        edges['Retail Density'].iloc[i]=df['Retail Density'].iloc[0]
    
    return edges                                           
                                            
def RiskFactorGraph(block_name,edges,nodes,path,column):

  '''
  This function plots the graph of the Risk Factors and saves it at the path
  It takes in the edges dataframe,nodes dataframe,the block name,the column to be plotted
  and the filepath where the image needs to be saved

  '''

  gdf = ox.geocoder.geocode_to_gdf("North Hills , Los Angeles")
  footprints=ox.geometries.geometries_from_place("North Hills Los Angeles", tags={"building":True})
  fig, ax = plt.subplots(figsize=(20, 20))
  gdf.plot(ax=ax, facecolor='black')
  edges.plot(ax=ax, linewidth=2, column=column, cmap='YlOrBr')
  nodes.plot(ax=ax, linewidth=0, facecolor='white')
  footprints.plot(ax=ax,facecolor='green')
  sm = plt.cm.ScalarMappable(cmap='YlOrBr', norm = matplotlib.colors.Normalize(vmin=edges[column].min(), vmax=edges[column].min()))
  sm.set_array([])
  fig.colorbar(sm, ax=ax)
  ax.set_title('Visualizing risks')
  fig.tight_layout()
  fig.savefig(path+block_name+'_'+column+'_.png')

