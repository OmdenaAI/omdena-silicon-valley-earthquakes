import altair as alt
import datetime
import folium
import geopandas as gpd
import geopy
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import plotly_express as px
import plotly.io as pio
import plotly.offline as pyo
import requests
import seaborn as sns
import streamlit as st
import psycopg2

from folium.features import DivIcon
from googletrans import Translator
from PIL import Image
from shapely.geometry import Point, LineString
from spacy import displacy
from spacy_streamlit import visualize_ner
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

from branca.element import Figure
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

ox.config(use_cache=True, log_console=True)
pyo.init_notebook_mode(connected=True)

st.set_page_config(
    page_title="California Earthquake Safe Path",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    """
    <style>
        .css-hby737, .css-17eq0hr, .css-qbe2hs {
            background-color:    #154360    !important;
            color: white !important;
        }
        div[role="radiogroup"] {
            color:white !important;
            margin-left:8%;
        }
        div[data-baseweb="select"] > div {
            
            color: black;
        }
        div[data-baseweb="base-input"] > div {
            background-color: #aab7b8 !important;
            color: black;
        }
        
        .st-cb, .st-bq, .st-aj, .st-c0{
            color: white !important;
        }


        .st-bs, .st-ez, .st-eq, .st-ep, .st-bd, .st-e2, .st-ea, .st-g9, .st-g8, .st-dh, .st-c0 {
            color: black !important;
        }

        .st-fg, .st-fi {
            background-color: #c6703b !important;
            color: white !important;
        }

       
        
        .st-g0 {
            border-bottom-color: #c6703b !important;
        }

        .st-fz {
            border-top-color: #c6703b !important;
        }

        .st-fy {
            border-right-color: #c6703b !important;
        }

        .st-fx {
            border-left-color: #c6703b !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown('<h1 style="margin-left:8%; color:white">California Earthquake Safe Path </h1>', unsafe_allow_html=True)

add_selectbox = st.sidebar.radio(
    "",
    ("Query", "Maps", "Configurations", "References")
)




if add_selectbox == 'Query':
    current_location = st.text_input('Current Location:') 

    if st.button('Search'):
        api_token = "<insert_your_mapbox_token_here>"
        
        def create_graph(loc, dist, transport_mode, loc_type="address"):
            """Transport mode = ‘walk’, ‘bike’, ‘drive’, ‘drive_service’, ‘all’, ‘all_private’, ‘none’"""
            if loc_type == "address":
                G = ox.graph_from_address(loc, dist=dist, network_type=transport_mode)    
            elif loc_type == "points":
                G = ox.graph_from_point(loc, dist=dist, network_type=transport_mode )

            return G
        
        G = create_graph("Northridge, California, USA", 2500, "drive")
        ox.plot_graph(G)
        
        G = ox.add_edge_speeds(G) #Impute
        G = ox.add_edge_travel_times(G) #Travel time

        # start = (57.715495, 12.004210)
        # end = (57.707166, 11.978388)

        start = (34.2546975,-118.5165975)
        end = (34.2272802, -118.5013579)

        start_node = ox.get_nearest_node(G, start)
        end_node = ox.get_nearest_node(G, end)# Calculate the shortest path
        route = nx.shortest_path(G, start_node, end_node, weight='travel_time')

        #Plot the route and street networks
        ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
        
        node_start = []
        node_end = []
        X_to = []
        Y_to = []
        X_from = []
        Y_from = []
        length = []
        travel_time = []

        for u, v in zip(route[:-1], route[1:]):
            node_start.append(u)
            node_end.append(v)
            length.append(round(G.edges[(u, v, 0)]['length']))
            travel_time.append(round(G.edges[(u, v, 0)]['travel_time']))
            X_from.append(G.nodes[u]['x'])
            Y_from.append(G.nodes[u]['y'])
            X_to.append(G.nodes[v]['x'])
            Y_to.append(G.nodes[v]['y'])
            
            
        df = pd.DataFrame(list(zip(node_start, node_end, X_from, Y_from, X_to, Y_to, length, travel_time)),
                  columns =["node_start", "node_end", "X_from", "Y_from", "X_to", "Y_to", "length", 
                            "travel_time"])
        
        def create_line_gdf(df):
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X_from, df.Y_from))
            gdf["geometry_to"] = [Point(xy) for xy in zip(gdf.X_to, gdf.Y_to)]
            gdf['line'] = gdf.apply(lambda row: LineString([row['geometry_to'], row['geometry']]), axis=1)

            line_gdf = gdf[["node_start","node_end","length","travel_time", "line"]].set_geometry('line')

            return line_gdf
        
        line_gdf = create_line_gdf(df)
        start = df[df["node_start"] == start_node]
        end = df[df["node_end"] == end_node]
        
        fig = px.scatter_mapbox(df, lon= "X_from", lat="Y_from", zoom=12, width=1000, height=800)
        fig.update_layout(font_size=16,  title={'xanchor': 'center','yanchor': 'top', 'y':0.9, 'x':0.5,}, 
                title_font_size = 24, mapbox_accesstoken=api_token, 
                          mapbox_style = "mapbox://styles/strym/ckhd00st61aum19noz9h8y8kw")
        fig.update_traces(marker=dict(size=6))
        
        st.write(fig)
        
        fig = px.scatter_mapbox(df, lon= "X_from", lat="Y_from", 
                        zoom=13, width=1000, height=800, animation_frame=df.index, mapbox_style="dark")
        fig.data[0].marker = dict(size = 12, color="black")
        fig.add_trace(px.scatter_mapbox(start, lon= "X_from", lat="Y_from").data[0])
        fig.data[1].marker = dict(size = 15, color="red")
        fig.add_trace(px.scatter_mapbox(end, lon= "X_from", lat="Y_from").data[0])
        fig.data[2].marker = dict(size = 15, color="green")
        fig.add_trace(px.line_mapbox(df, lon= "X_from", lat="Y_from").data[0])

        fig.update_layout(font_size=16,  title={'xanchor': 'center','yanchor': 'top', 'y':0.9, 'x':0.5,}, 
                title_font_size = 24, mapbox_accesstoken=api_token)
        st.write(fig)
        
#         geolocator = Nominatim(user_agent="tutorial")
#         location = geolocator.geocode(current_location).raw

#         my_expander = st.beta_expander('OSM Details', True)
#         with my_expander:

#             st.markdown('<span><b>{} Details:</b></span>:'.format(current_location), unsafe_allow_html=True)
#             st.markdown('<span><b>Latitude</b></span>:   {}'.format(location['lat']), unsafe_allow_html=True)
#             st.markdown('<span><b>Longtitude</b></span>: {}'.format(location['lon']), unsafe_allow_html=True)
#             st.markdown('<span><b>OSM ID</b></span>: {}'.format(location['osm_id']), unsafe_allow_html=True)
#             st.markdown('<span><b>Bounding Box</b></span>: {}'.format(location['boundingbox']), unsafe_allow_html=True)
#             st.markdown('<span><b>Place ID</b></span>: {}'.format(location['place_id']), unsafe_allow_html=True)
#             st.markdown('<span><b>Display Name</b></span>: {}'.format(location['display_name']), unsafe_allow_html=True)
#             st.markdown('<span><b>Type</b></span>: {}'.format(location['type']), unsafe_allow_html=True)
#             st.markdown('<span><b>Class</b></span>: {}'.format(location['class']), unsafe_allow_html=True)

        

        fig2=Figure(width=550,height=350)
        m2=folium.Map(location=[34.2819, 118.4390], zoom_start=3)
        fig2.add_child(m2)
        folium.TileLayer('Stamen Terrain').add_to(m2)
        folium.TileLayer('Stamen Toner').add_to(m2)
        folium.TileLayer('Stamen Water Color').add_to(m2)
        folium.TileLayer('cartodbpositron').add_to(m2)
        folium.TileLayer('cartodbdark_matter').add_to(m2)
        folium.LayerControl().add_to(m2)

#         loc = [(40.720, -73.993),
#              (40.721, -73.996)]

#         folium.PolyLine(loc, color='red',weight=15, opacity=0.8).add_to(m2)

#         folium.Marker(location=[34.2819, 118.4390], popup='Default popup Marker1',
#                       tooltip='Click here to see Popup').add_to(m2)
#         folium_static(m2)

        
        df = pd.read_csv('Worldwide-Earthquake-database.csv')

        df['LONGITUDE'] = pd.to_numeric(df.LONGITUDE, errors='coerce')
        df['LATITUDE'] = pd.to_numeric(df.LATITUDE, errors='coerce')# drop rows with missing lat, lon, and intensity
        df.dropna(subset=['LONGITUDE', 'LATITUDE', 'INTENSITY'], inplace=True)# convert tsunami flag from string to int
        df['FLAG_TSUNAMI'] = [1 if i=='Yes' else 0 for i in df.FLAG_TSUNAMI.values]

        from streamlit_keplergl import keplergl_static
        from keplergl import KeplerGl

        map_1 = KeplerGl(height=800)
        map_1.add_data(data=df, name="earthquakes")
        keplergl_static(map_1)



