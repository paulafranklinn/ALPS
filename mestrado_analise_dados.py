
## importas bibliotecas

import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import matplotlib.patheffects as PathEffects
from turtle import bgcolor
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing, feature_selection
import matplotlib.colors as mcolors
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from time import time
import gzip
import os
import plotly as py
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
import scipy.cluster.hierarchy as sch
from sklearn import metrics, datasets
from sklearn.datasets import make_blobs
import matplotlib.cm as cm
import imageio
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score


## CLUSTERIZAÇÃO: KMEANS
def kMeansRes(scaled_data, k, alpha_k=0.02):
    '''
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns 
    -------
    scaled_inertia: float
        scaled inertia value for current k           
    '''
    
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia

def chooseBestKforKMeansParallel(scaled_data, k_range):

    '''
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k_range: list of integers
        k range for applying KMeans
    Returns 
    -------
    best_k: int
        chosen value of k out of the given k range.
        chosen k is k with the minimum scaled inertia value.
    results: pandas DataFrame
        adjusted inertia value for each k in k_range
    '''
    
    ans = Parallel(n_jobs=-1,verbose=10)(delayed(kMeansRes)(scaled_data, k) for k in k_range)
    ans = list(zip(k_range,ans))
    results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k, results

def chooseBestKforKMeans(scaled_data, k_range):
    ans = []
    for k in k_range:
        scaled_inertia = kMeansRes(scaled_data, k)
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k, results

def bestkclust(data):
    dtmtx = np.asarray(data).astype(float)
    mms = MinMaxScaler()
    k_range=range(2,20)
    scaleddata = mms.fit_transform(dtmtx)
    ### MELHOR K:
    melhor_k = chooseBestKforKMeans(scaleddata, k_range)
    num_clusters = melhor_k[0]
    # Perform K-Means clustering on the data
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    # Get the cluster labels for each data point
    labels = kmeans.labels_
    df = data
    # Add the cluster labels to the original DataFrame
    df["Cluster"] = labels
    return df, num_clusters, labels

## PCA
def pca_dataset(dataset, dimensions):
    pca_features= PCA(n_components=dimensions)
    pc_features = pca_features.fit_transform(dataset)
    principalComponents_features = pd.DataFrame(pc_features)
    pca_features_all = PCA(n_components= min(np.shape(dataset)))
    principalComponents_features_all = pca_features_all.fit_transform(dataset)
    x=0
    explained_variance = [(x+pca_features.explained_variance_[i]/np.sum(pca_features_all.explained_variance_))*100 for i in range(dimensions)]
    explained_variance_sum = np.sum(explained_variance)
    return principalComponents_features, explained_variance, explained_variance_sum

def plotar_grafico(rep_n,media, desvio_padrao,r):
    ciclos = np.arange(1, len(media) + 1)

    plt.figure(figsize=(10, 6))
    plt.errorbar(ciclos, media, yerr=desvio_padrao, fmt='o-', capsize=5)
    plt.title(f'Replica {rep_n}, corr = {r} - RMSE e Desvio Padrão por Ciclo', plt.savefig(f'replica{rep_n}/RMSE_SD_replica_{rep_n}.png', transparent = False,  facecolor = 'white'))
    plt.xlabel('Ciclo')
    plt.ylabel('Valor')
    plt.grid(True)
    plt.show()

def red_tsne(data,dimension,prp):
    X = np.array(data)
    tsn = TSNE(n_components= dimension,perplexity= prp,random_state= 42,n_iter= 5000,n_jobs= -1)
    X_embedded = tsn.fit_transform(X)
    X_embedded.shape
    dt_tsne_all = pd.DataFrame(X_embedded)
    return dt_tsne_all
    
testando = plotar_dados_3d('/home/paula/Documentos/Mestrado/dados_050224/peptide_machine/analise_dados/',"TSNE",2)

def plotar_dados_3d(directory_set,method,dimension):

    # directory_set -> diretório contendo arquivos csv
    # method: "TSNE" ou "PCA"
    
    os.chdir(directory_set)
    arquives_csv = os.listdir()
    n_replicas = len(arquives_csv)

    for j in range(n_replicas):

        os.mkdir(f'replica{j}')
        ## Seleção do arquivo csv
        dados_analise = pd.read_csv(f"df_dados_gerais_replica_{j}.csv")
        dados_analise_df = pd.DataFrame(dados_analise)
        dados_analise_df = dados_analise_df.iloc[:,1:]

             ## Normalizando dados
        dados_descritores = dados_analise_df.iloc[-1000:,7:]
        scaler = StandardScaler()
        scaler.fit(dados_descritores)
        dados_descritores_norm = scaler.transform(dados_descritores)
        dados_descritores_norm = pd.DataFrame(dados_descritores_norm)

                ## TSNE
        if method == "TSNE":
            coord = red_tsne(dados_descritores_norm,dimension,500)
            coord_df = pd.DataFrame(coord)
            coord_df.to_csv(f"replica{j}/coord_TSNE_replica{j}.csv")

                ## PCA
        else:
            coord, var_exp, soma_var = pca_dataset(dados_descritores_norm,dimension)
            print("doing PCA")

        a = pd.DataFrame()

        for p in range(0,1050,50):
            an = coord_df.iloc[:p,:]
            a = pd.concat([a,an])       

        coord_dff = a.reset_index()
        coord_dff = coord_dff.iloc[:,1:]
        ciclos_ = dados_analise_df["Ciclo"].unique()
        ciclos = len(ciclos_)       

        for n in range(ciclos):
            if dimension == 3:
                        ## Criando dataframe para análise
                df = pd.DataFrame({'PC1': coord_dff.iloc[:,0],
                                        'PC2': coord_dff.iloc[:,1],
                                        'PC3': coord_dff.iloc[:,2],
                                        'Ciclo': dados_analise_df.iloc[:,4],
                                        'Treino_teste': dados_analise_df.iloc[:,3],
                                        'Observado': dados_analise_df.iloc[:,5],
                                        "Predito": dados_analise_df.iloc[:,6],
                                        'Erro': dados_analise_df.iloc[:,7].abs()})

                dados_analise_dff = df[df['Ciclo'] == n]

                            # Plotar o gráfico 3D com coloração por ciclo e tamanho por erro
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(dados_analise_dff['PC1'], dados_analise_dff['PC2'], dados_analise_dff['PC3'], c=dados_analise_dff['Erro'].astype(float),cmap="seismic", alpha=0.6)

                cbar = plt.colorbar(scatter)

                            # Set the desired limits for legend labels
                min_limit = min(df["Erro"])  # Replace with your desired minimum value
                max_limit = max(df["Erro"])  # Replace with your desired maximum value
                cbar.set_ticks([min_limit, max_limit])
                cbar.set_label('Erro')

                            # Adicionar barra de cores
                lim_x = (min(df['PC1']),max(df['PC1']))
                lim_y = (min(df['PC2']),max(df['PC2']))
                lim_z = (min(df['PC3']),max(df['PC3']))

                ax.set_xlim(lim_x)
                ax.set_ylim(lim_y)
                ax.set_zlim(lim_z)

                            # Configurar rótulos dos eixos
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')

                plt.title(f'{method}| replica {j} | Ciclo {k}', fontsize=14), plt.savefig(f'replica{j}/replica_{j}_{method}_Ciclo_{n}.png', transparent = False,  facecolor = 'white')
                            #plt.show()plotar_dados_3d(directory_set,method)

            else:
                            ## Criando dataframe para análise
                df = pd.DataFrame({'PC1': coord_dff.iloc[:,0],
                                    'PC2': coord_dff.iloc[:,1],
                                    'Ciclo': dados_analise_df["Ciclo"],
                                    'Treino_teste': dados_analise_df['Treino_ou_teste'],
                                    'Erro': dados_analise_df['erro_modelo'].abs()})
                
                data_dff = df[df['Ciclo'] == n]

                fig = plt.figure(figsize=(8, 8))
                sns.scatterplot(df, x=data_dff['PC1'],y=data_dff['PC2'], c=data_dff['Erro'].astype(float),cmap="plasma_r", legend=False, alpha=1.0)

                plt.xlim(min(df.iloc[:,0]), max(df.iloc[:,0]))
                plt.ylim(min(df.iloc[:,1]), max(df.iloc[:,1]))

                fig.suptitle(f'KMeans clustering after {method} in VHSE data')
                            #display(fig)
                fig.savefig(f"replica{j}/replica_{j}_{method}_Ciclo_{n}.png", dpi=600)

                
        frames = []
        for i in range(ciclos):
            image = imageio.v2.imread(f'replica{j}/replica_{j}_{method}_Ciclo_{i}.png')
            frames.append(image)
            
        imageio.mimsave(f'replica{j}/gif_replica_{j}.gif', # output gif
        frames,          # array of input frames
        duration = 2000,
        loop = 1)

    return arquives_csv


for u in range(5): 
    dados_analise = pd.read_csv(f"df_dados_gerais_replica_{u}.csv")
    dados_analise_df = pd.DataFrame(dados_analise)

    list_RMSE = []
    list_desv = []

    df = pd.DataFrame({'Ciclo': dados_analise_df.iloc[:,4],
                        'Treino_teste': dados_analise_df.iloc[:,3],
                        'Observado': dados_analise_df.iloc[:,5],
                        'Predito': dados_analise_df.iloc[:,6],
                        'Erro': dados_analise_df.iloc[:,7].abs()}) 

    for l in range(20):        
        fd_RMSE = df[(df["Ciclo"] == l) & (df["Treino_teste"] == 1)]

        plt.xlabel("Rosetta")
        plt.ylabel('Modelo')
        plt.scatter(fd_RMSE["Observado"], fd_RMSE["Predito"], alpha=0.5)
        plt.savefig(f'replica{u}/rosetta_vs_replicata_{u}.png', dpi = 300)  

        rootMeanSqErr = np.sqrt(metrics.mean_squared_error(fd_RMSE["Observado"], fd_RMSE["Predito"]))
        r = np.corrcoef(fd_RMSE["Observado"], fd_RMSE["Predito"])
        list_RMSE.append(rootMeanSqErr)
        ann = np.std(fd_RMSE["Erro"])
        list_desv.append(ann)

    plotar_grafico(u,list_RMSE,list_desv,r)

# Import statistics Library
from statistics import variance
a = []
for y in range(5):
    dados_analise = pd.read_csv(f"//home/paula/Documentos/Mestrado/dados_050224/peptide_machine/analise_dados/df_dados_gerais_replica_{y}.csv")
    dados_analise_df = pd.DataFrame(dados_analise)

    df = pd.DataFrame({'Ciclo': dados_analise_df.iloc[:,4],
                        'Treino_teste': dados_analise_df.iloc[:,3],
                        'Observado': dados_analise_df.iloc[:,5],
                        'Predito': dados_analise_df.iloc[:,6],
                        'Erro': dados_analise_df.iloc[:,7].abs()}) 
    for l in range(20):
        fd_RMSE = df[(df["Ciclo"] == l)]
        an = variance(fd_RMSE["Observado"])
        a.append(ann)

r = np.corrcoef(fd_RMSE["Observado"], fd_RMSE["Predito"])


frames = []
for i in range(100):
    image = imageio.v2.imread(f'replica{j}/replica_{j}_TSNE_Ciclo_{i}.png')
    frames.append(image)
            
imageio.mimsave(f'replica{j}/gif_replica_{j}.gif', # output gif
frames,          # array of input frames
duration = 2000,
loop = 1)

for i in range(0,1000,50):

    fd = df.iloc[:i,:]

    fig = plt.figure(figsize=(8, 8))
    sns.scatterplot(fd, x=fd['PC1'],y=fd['PC2'], c=fd['Erro'].astype(float),cmap="plasma_r", legend=False, alpha=1.0)

    plt.xlim(min(df['PC1']), max(df['PC1']))
    plt.ylim(min(df['PC2']), max(df['PC2']))

    fig.suptitle(f'KMeans clustering after TSNE in VHSE data')
    #display(fig)
    fig.savefig(f"replica0/replica_0_TSNE_Ciclo_{i}.png", dpi=600)

frames = []
for i in range(0,1000,50):
    image = imageio.v2.imread(f'/home/paula/Documentos/Mestrado/dados_050224/peptide_machine/analise_dados/replica0//replica_0_TSNE_Ciclo_{i}.png')
    frames.append(image)
            
imageio.mimsave(f'/home/paula/Documentos/Mestrado/dados_050224/peptide_machine/analise_dados/replica0/gif_replica_0.gif', # output gif
frames,          # array of input frames
duration = 800,
loop = 2)
