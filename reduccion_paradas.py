# Imports

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

from scipy.sparse import coo_matrix, save_npz, load_npz, csgraph

from geopy.distance import geodesic

from sklearn.neighbors import radius_neighbors_graph
from sklearn.cluster import AgglomerativeClustering

# %% Configs

pio.renderers.default='browser'
# pio.renderers.default='svg'

# %% Data

path_data_colectivos = "./../data/colectivos-gtfs/"

stops_completo = pd.read_table(path_data_colectivos+"stops.txt", sep=',')

# %% Plots

def grafo_de_conectividad(stops, conectividad):
    """
    Plotea todas las paradas y todas las aristas de la conectividad
    """
    coo = conectividad.tocoo()
    lats = []
    lons = []

    for i, j in zip(coo.row, coo.col):
        if i >= j:
            continue

        lats.extend([
            stops.loc[i, "stop_lat"],
            stops.loc[j, "stop_lat"],
            None
        ])
        lons.extend([
            stops.loc[i, "stop_lon"],
            stops.loc[j, "stop_lon"],
            None
        ])

    fig = px.scatter_map(stops, lat="stop_lat", lon="stop_lon", zoom=10)
    fig.add_trace(
        go.Scattermap(
            lat=lats,
            lon=lons,
            mode="lines",
            line=dict(width=1, color="rgba(0,0,0,0.3)"),
            name="conectividad"
        )
    )
    fig.show()



# %% Matriz de distancias

STOPS = None
CONECTIVIDAD = None

def init_worker(stops, conectividad):
    """
    Asigna stops y conectividad como variables globales para no tener que
    cargarlas en todas las iteraciones y no se puede pasar por referencia a los
    procesos.
    """
    global STOPS, CONECTIVIDAD
    STOPS = stops
    CONECTIVIDAD = conectividad
 

def calcular_fila(i):
    stop1 = STOPS.loc[i]
    print(i)
    
    resultados = []
    for j in CONECTIVIDAD.getcol(i).nonzero()[0]:
        if i<j: continue
        
        stop2 = STOPS.loc[j]
        dist = geodesic((stop1.stop_lat, stop1.stop_lon),
                        (stop2.stop_lat, stop2.stop_lon)).meters
    
        resultados.append((i, j, dist))

    return resultados


def calcular_matriz_distancia(stops, conectividad, save_path=None):
    rows, cols, data = [], [], []

    with ProcessPoolExecutor(max_workers=12,
                             initializer=init_worker,
                             initargs=(stops, conectividad)
                             ) as executor:
        
        for resultados in executor.map( calcular_fila, stops.index ):
            for i, j, dist in resultados:
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([dist, dist])

    dist_sparse = coo_matrix((data, (rows, cols)),
                             shape=(len(stops), len(stops))).tocsr()
    
    if save_path:
        save_npz(save_path, dist_sparse)
        
    return dist_sparse


# %% 
"""
Calculo un subgrafo de conectividad de las paradas para distancias de menos de 
0.01° (~1km) en lat long con distancia euclidea
"""
conectividad_completo = radius_neighbors_graph(
    stops_completo[['stop_lat', 'stop_lon']], radius=.01, mode='connectivity',
    include_self=False, n_jobs=10)

save_npz("conectividad_completo.npz", conectividad_completo)

dist_matrix_completo = calcular_matriz_distancia(
                            stops_completo, conectividad_completo,
                            "matriz_distancia_completo.npz")

# %%
"""
Reduzco la matriz de distancias uniendo todas las paradas que están a menos de
50 metros 2 a 2 y tomando una de las dos posiciones (es más rápido que tomar 
el centroide porque no hay que recalcular distancias)
"""
def reducir_stops(stops, dist_sparse, umbral=50):
    coo = dist_sparse.tocoo()

    edges = sorted(
        zip(coo.row, coo.col, coo.data),
        key=lambda x: x[2]
    )

    used = set()
    pairs = []

    for i, j, d in edges:
        if d < umbral and i not in used and j not in used:
            pairs.append((i, j))
            used.update([i, j])
            
    if not pairs:
        return stops, dist_sparse, False
    
    
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
            "members": stops.loc[i, "members"] + stops.loc[j, "members"]
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
    
    stops_nuevo = pd.DataFrame(new_rows).reset_index(drop=True)
    
    # --- reconstruir matriz ---
    rows, cols, data = [], [], []

    for i, j, d in zip(coo.row, coo.col, coo.data):
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

    dist_nueva = coo_matrix(
        (data, (rows, cols)),
        shape=(len(stops_nuevo), len(stops_nuevo))
    ).tocsr()

    return stops_nuevo, dist_nueva, True




stops_reducido = stops_completo[["stop_id","stop_lat","stop_lon"]].copy()
stops_reducido["members"] = stops_reducido.apply(
                                lambda x: [x["stop_id"]], axis=1)
dist_reducido = load_npz("matriz_distancia_completo.npz")

it = 0
while True:
    stops_reducido, dist_reducido, cambio = reducir_stops(stops_reducido,
                                                          dist_reducido)
    print(f"iter {it}: {len(stops_reducido)} nodos")
    it += 1
    if not cambio:
        break

stops_reducido.to_csv('stops_reducido.csv')
save_npz("matriz_distancia_reducida.npz", dist_reducido)


# %%

conectividad_reducido = radius_neighbors_graph(
    stops_reducido[['stop_lat', 'stop_lon']], .01,
    mode='connectivity', include_self=False, n_jobs=10)

save_npz("conectividad_reducido.npz", conectividad_reducido)


# %% Clustering Reducido

conectividad_reducido = load_npz("matriz_distancia_reducida.npz")
dist_reducido = load_npz("matriz_distancia_reducida.npz")


clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=500, 
                metric='precomputed',
                connectivity=conectividad_reducido,
                linkage='complete'
            ).fit(dist_reducido.toarray())


fig = px.scatter_map(stops_reducido, lat="stop_lat", lon="stop_lon", zoom=10,
                     color=clustering.labels_)
fig.show()

