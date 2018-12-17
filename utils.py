import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
from sklearn.cluster import KMeans
import numpy as np
from random import shuffle


def fetch_od(dataset: pd.DataFrame):
    order_series = dataset['orderid'].drop_duplicates()

    result = []
    with tqdm(total=order_series.shape[0], desc='Fetching O-D point pairs: ') as tqdm_iter:
        for orderid, group in dataset.groupby('orderid'):
            group = group.sort_values('timestamp')
            result.append(group.iloc[0])
            result.append(group.iloc[-1])
            tqdm_iter.update(1)
    result = pd.DataFrame(result)
    result.index = range(result.shape[0])

    return result


def add_od_to_graph(graph: nx.Graph, o, d):
    try:
        graph[o][d]['weight'] += 1
    except KeyError:
        graph.add_nodes_from([o, d])
        graph.add_edge(o, d, weight=1)
        
        
def add_dataset_to_graph(dataset: pd.DataFrame, graph):
    with tqdm(total=dataset.shape[0] / 2, desc='Adding dataset to graph: ') as tqdm_iter:
        for orderid, group in dataset.groupby('orderid'):
            add_od_to_graph(graph, group.iloc[0]['cluster'], group.iloc[1]['cluster'])
            tqdm_iter.update(1)
         
        
def draw_timeslot_graph(graph, cluster_centers, od_set, slot_range,\
                        point_color='red', point_style='D', arrow_color='#f9f568',\
                        figsize=(10, 10), dpi=100, thre=50, norm=100, save_file=None):
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
        
    for edge in graph.edges():
        start, end = edge[0:2]
        if start == end:
            continue
        
        weight = sum(graph[start][end]['weight'][slot_range[0]:(slot_range[1]+1)])
        if weight < thre:
            continue

        x1 = cluster_centers[start][0]
        y1 = cluster_centers[start][1]
        x2 = cluster_centers[end][0]
        y2 = cluster_centers[end][1]
        width = (weight-thre)/norm

        ax.annotate('', xy = (x2, y2),xytext = (x1, y1), fontsize = 7, color=arrow_color, 
                    arrowprops=dict(edgecolor='black', facecolor=arrow_color, shrinkA=0, 
                                    shrinkB=0, width=width, headwidth=max(width*2, 4), alpha=0.8))
    
    plt.scatter(od_set['longitude'], od_set['latitude'], alpha=0.1, s=1, c='#3677af')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=point_color, marker=point_style, s=40)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        

def add_timeslot_weight_to_graph(dataset, graph, slot_length=1):
    def add_od_to_graph_with_timeslot(graph, o, d, slot_index):
        try:
            weight = graph[o][d]['weight']
            graph[o][d]['weight'][slot_index] += 1
        except KeyError:
            graph.add_nodes_from([o, d])
            graph.add_edge(o, d)
            graph[o][d]['weight'] = [0] * int(24 / slot_length)
            graph[o][d]['weight'][slot_index] += 1

    with tqdm(total=int(dataset.shape[0] / 2), desc='Adding dataset to graph: ') as pbar:
        for orderid, group in dataset.groupby('orderid'):
            slot_index = int(group.iloc[0]['timestamp'].hour / slot_length)
            add_od_to_graph_with_timeslot(graph, group.iloc[0]['cluster'], group.iloc[1]['cluster'], 
                                         slot_index)
            pbar.update(1)

        
def construct_slot_graph(file_dir, kmeans=None, n_clusters=200, slot_length=1):
    one_day_data = pd.read_csv(file_dir, names=['driverid', 'orderid', 'timestamp', 'longitude', 'latitude'])
    one_day_data['timestamp'] = pd.to_datetime(one_day_data['timestamp'], unit='s') + pd.to_timedelta(8, unit='h')
    
    od_set = fetch_od(one_day_data)
    
    if kmeans is None:
        kmeans = KMeans(n_clusters=n_clusters, n_jobs=48).fit(od_set[['longitude', 'latitude']])
    od_set['cluster'] = kmeans.labels_
    
    slot_graph = nx.DiGraph()
    add_timeslot_weight_to_graph(od_set, slot_graph, slot_length=slot_length)
    
    return slot_graph, od_set, kmeans


def show_divided_areas(od_set, centers, figsize=(15, 15), dpi=150, text=False):
    clusters = []
    groups = []
    for cluster, group in od_set.groupby('cluster'):
        clusters.append(cluster)
        groups.append(group)
    group_pair = list(zip(clusters, groups))
    shuffle(group_pair)
    
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    for cluster, group in group_pair:
        center = centers[cluster]
        
        plt.scatter(group['longitude'], group['latitude'], s=1, alpha=0.5)
        plt.scatter(center[0], center[1], s=40, marker='D', color='black', alpha=0.5)
        
        if text:
            ax.annotate(str(cluster), xy=(center[0], center[1]), 
                        xytext=(center[0]+0.00025, center[1]+0.00025), fontsize=8, alpha=0.5)
    plt.show()
    return
