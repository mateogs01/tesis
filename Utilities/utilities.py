from typing import Type
from concurrent.futures import ProcessPoolExecutor

# import numpy as np
import pandas as pd

# import plotly.io as pio
# import plotly.express as px
# import plotly.graph_objects as go

from scipy.sparse import coo_matrix, csr_matrix, save_npz, csgraph

from geopy.distance import geodesic

from sklearn.neighbors import radius_neighbors_graph
from sklearn.cluster import AgglomerativeClustering

# %%


def connectivityMatrix(stops, radius, n_processes, save_path=None):
    """
    Computes the conectivy matrix of all the stops given a radius in
    latitude and longitud degrees.
    """
    
    connectivity = radius_neighbors_graph(
        stops[['stop_lat', 'stop_lon']],
        radius=radius,
        mode='connectivity',
        include_self=False,
        n_jobs=n_processes)
    
    if save_path:
        save_npz(save_path, connectivity)
    
    return connectivity


def _initWorkerDistMatrix(stops, connectivity):
    global STOPS, CONNECTIVITY
    STOPS = stops
    CONNECTIVITY = connectivity


def _rowDistMatrix(i):
    print(i)

    stop1 = STOPS.loc[i]

    res = []
    for j in CONNECTIVITY.getcol(i).nonzero()[0]:
        if i < j:
            continue

        stop2 = STOPS.loc[j]
        dist = geodesic(
            (stop1.stop_lat, stop1.stop_lon),
            (stop2.stop_lat, stop2.stop_lon)
        ).meters

        res.append((i, j, dist))

    return res    


def distanceMatrix(stops, connectivity, n_processes, save_path=None):
    """
    Computes the sparse distance matrix of all stops given the connectivity
    matrix.
    """
    
    STOPS = None
    CONNECTIVITY = None
    
    rows, cols, data = [], [], []

    with ProcessPoolExecutor(
        max_workers=n_processes,
        initializer=_initWorkerDistMatrix,
        initargs=(stops, connectivity)
    ) as executor:

        for res in executor.map(_rowDistMatrix, stops.index):
            for i, j, dist in res:
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([dist, dist])

    sparse_matrix = coo_matrix(
        (data, (rows, cols)),
        shape=(len(stops), len(stops))
    ).tocsr()

    if save_path:
        save_npz(save_path, sparse_matrix)

    return sparse_matrix
    
# %%

def mergeStops(stops, threshold, method='euclidean',
               sparse_dist_matrix=None, centroids=True, n_processes=1,
               MAX_ITER=10, save_path_dist_matrix=None, save_path_stops=None):
    """
    Merges stops that are closer than a given threshold. The list of stops_id
        merge are saved in the members column of the dataframe returned.
    If method euclidean is chosen, the threshold is interpreted as degrees in
        the gps coordinates. The centroids parameter is used.
    If method geodesic is chosen, the threshold is interpretes as meters. A 
        distance matrix is required.
    """
    
    stops["members"] = stops.apply(lambda x: [x["stop_id"]], axis=1)

    it = 0
    changed = True
    
    print(f"iter {it}: {len(stops)} total nodes")
    
    while changed or it>MAX_ITER:
        if method == 'euclidean':
            stops, changed = _mergeStopsIterationEuclidean(
                                stops, threshold, n_processes, centroids)
        
        elif method == 'geodesic':    
            stops, dist_reduced, changed = _mergeStopsIterationGeodesic(
                                stops, sparse_dist_matrix, threshold)
        
        else:
            raise Exception("invalid method")
        
        it += 1
        print(f"iter {it}: {len(stops)} total nodes")


    if save_path_dist_matrix:
        save_npz(save_path_dist_matrix, dist_reduced)
    
    if save_path_stops:
        stops.to_pickle(save_path_stops)
    
    
    if method == 'euclidean':
        return stops
    
    elif method == 'geodesic':   
        return stops, dist_reduced


def _mergeStopsIterationGeodesic(stops, dist_matrix, connectivity, threshold):
    """
    Computes one merge iteration based on the geodesic distance measured in
    meters. Needs a distance matrix and connectivity to work.
    """
    raise Exception("funcion no implementada")


def _mergeStopsIterationEuclidean(stops, threshold, n_processes, centroids=True):
    """
    Computes one merge iteration based on the euclidean distance measured in
    degrees.
    If centroids is true, merging an edge will cause to recalculate the position of
    the node in the center of both. Else, it will use one of the two nodes as its position.
    """
    
    connectivity = connectivityMatrix(stops, threshold, n_processes)

    pairs = []
    used = set()
    
    for i in connectivity.indices:
        for j in connectivity[i].indices:
            if i not in used and j not in used:
                pairs.append((i, j))
                used.update([i, j])
            
    if not pairs:
        return stops, False
    
    
    # --- construir nuevos nodos ---
    new_rows = []
    old_to_new = {}
    new_id = 0
    
    for i, j in pairs:
        if centroids:
            new_rows.append({
                "stop_lat": (stops.loc[i, "stop_lat"] + stops.loc[j, "stop_lat"])/2,
                "stop_lon": (stops.loc[i, "stop_lon"] + stops.loc[j, "stop_lon"])/2,
                "members":  stops.loc[i, "members"]   + stops.loc[j, "members"]
            })
        else:
            new_rows.append({
                "stop_lat": stops.loc[i, "stop_lat"],
                "stop_lon": stops.loc[i, "stop_lon"],
                "members":  stops.loc[i, "members"]
                            + stops.loc[j, "members"]
            })
        old_to_new[i] = old_to_new[j] = new_id
        new_id += 1
    
    
    for i in stops.index:
        if i not in used:
            new_rows.append(stops.loc[i].to_dict())
            old_to_new[i] = new_id
            new_id += 1
    
    stops_reduced = pd.DataFrame(new_rows).reset_index(drop=True)
    
    return stops_reduced, True



def _mergeStopsIterationOld(stops, sparse_dist_matrix, threshold):
    coo_dist_matrix = sparse_dist_matrix.tocoo()

    edges = sorted(
        zip(coo_dist_matrix.row,
            coo_dist_matrix.col,
            coo_dist_matrix.data),
        key=lambda x: x[2]
    )

    used = set()
    pairs = []

    for i, j, d in edges:
        if d < threshold and i not in used and j not in used:
            pairs.append((i, j))
            used.update([i, j])
            
    if not pairs:
        return stops, sparse_dist_matrix, False
    
    
    # --- construir nuevos nodos ---
    new_rows = []
    old_to_new = {}
    representative = {}
    new_id = 0

    for i, j in pairs:
        rep = i
        new_rows.append({
            "stop_lat": stops.loc[rep, "stop_lat"],
            "stop_lon": stops.loc[rep, "stop_lon"],
            "members":  stops.loc[i, "members"]
                        + stops.loc[j, "members"]
        })
        old_to_new[i] = old_to_new[j] = new_id
        representative[i] = rep
        representative[j] = rep
        new_id += 1
    
    
    for i in stops.index:
        if i not in used:
            new_rows.append(stops.loc[i].to_dict())
            old_to_new[i] = new_id
            representative[i] = i
            new_id += 1
    
    stops_reduced = pd.DataFrame(new_rows).reset_index(drop=True)
    
    # --- reconstruir matriz ---
    rows, cols, data = [], [], []

    for i, j, d in zip(coo_dist_matrix.row,
                       coo_dist_matrix.col,
                       coo_dist_matrix.data):
        if i >= j:
            continue

        ni, nj = old_to_new[i], old_to_new[j]
        if ni == nj:
            continue
        
        if representative[i] != i or representative[j] != j:
            continue

        rows.extend([ni, nj])
        cols.extend([nj, ni])
        data.extend([d, d])

    sparse_dist_matrix_reduced = coo_matrix(
        (data, (rows, cols)),
        shape=(len(stops_reduced), len(stops_reduced))
    ).tocsr()

    return stops_reduced, sparse_dist_matrix_reduced, True




def stopsClustering(sparse_dist_matrix, connectivity, threshold):
    dist_expanded = sparse_dist_matrix.toarray()
    dist_expanded[dist_expanded==0] = 1e10
        
    clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    metric='precomputed',
                    connectivity=connectivity,
                    linkage='complete',
                    compute_full_tree=True
                ).fit(dist_expanded)

    return clustering


    


