"""
TODO:
    - Implementar MergeStops con distancia geodesica
    - Arreglar MergeStops que ahora retorna distintas cosas según el método que se usa

IDEAS:

"""

import pandas as pd
from scipy.sparse import load_npz

import plotly.io as pio
# import plotly.express as px
# import plotly.graph_objects as go


from Utilities import utilities as ut
# %% Configs
N_PROCESSES = 4

pio.renderers.default='browser'
# pio.renderers.default='svg'

path_data_colectivos = "./../data/colectivos-gtfs/"


# %%

# Archivos originales
stops_all   = pd.read_table(path_data_colectivos+"stops.txt", sep=',')
stop_times  = pd.read_table(path_data_colectivos+"stop_times.txt", sep=',')
trips_all   = pd.read_table(path_data_colectivos+"trips.txt", sep=',')


# %%

# Archivos generados en este código
stops_reduced = pd.read_pickle('stops_reducido.pkl')
connectivity_reduced = load_npz("matriz_distancia_reducida.npz")
sparse_distance_matrix_reduced = load_npz("matriz_distancia_reducida.npz")


# %%
"""
Calculo un subgrafo de conectividad de las paradas para distancias de menos de 
0.01° (~1km) en lat long con distancia euclidea
"""

connectivity_all = ut.connectivityMatrix(stops_all, 0.01, N_PROCESSES, "data/connectivity_all.npz")

# dist_all = ut.distanceMatrix(stops_all, connectivity_all, N_PROCESSES, "data/matriz_distancia_completo.npz")


# %%
"""
Reduzco la matriz de distancias uniendo todas las paradas que están a menos de
50 metros 2 a 2 y tomando una de las dos posiciones (es más rápido que tomar 
el centroide porque no hay que recalcu lar distancias)
"""

# stops_reduced, dist_reduced = ut.mergeStops(stops_all, dist_all, 50, "matriz_distancia_reducida.npz")
stops_reduced = ut.mergeStops(stops_all, 3e-3, n_processes=4, save_path_stops="data/stops_reduced.pkl")

connectivity_reduced = ut.connectivityMatrix(stops_reduced, .1, N_PROCESSES, "data/connectivity_reduced.npz")

dist_reduced = ut.distanceMatrix(stops_reduced, connectivity_reduced, N_PROCESSES, "data/dist_matrix_reduced.npz")



# %%
for dist_threshold in [300, 500, 1000, 1500, 2500, 5000]:
    print(dist_threshold)
    clustering = ut.stopsClustering(dist_threshold)
    
    stops_reduced[f"cluster_{dist_threshold}"] = clustering.labels_
    

