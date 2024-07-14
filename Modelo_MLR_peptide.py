
## bibliotecas
#import numpy as np
import pandas as pd
from sklearn.utils import column_or_1d

## Pre-processamento dos dados
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import pickle

## Algoritmos
from sklearn.linear_model import LassoCV
#from xgboost import XGBRegressor
#from sklearn.neural_network import MLPRegressor

## analise dos modelos
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
## Pre-processamento dos dados
import logging


# Load the data from a CSV file
#csv_file = "/home/papaula/Documentos/Mestrado//Peptides_as_descriptors_final_withdGandMethods_v02.csv"
#data = pd.read_csv(csv_file)
#print(data)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def pre_process(data,i):

    num_seed = (42+i)

    dataset = data
    
    ## dataset de treino e teste
    X = dataset.iloc[:,:-2]
    y = dataset['dG']
    
    
    # Split the data into training and testing sets with a random split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=num_seed)
    
    erro_df = pd.DataFrame(x_test)

    ## Selecionar index dos dados de teste e treino

    dados_gerais_df = pd.DataFrame(dataset.iloc[:,-1]) ## dados das sequencias do dataset
    dados_gerais_df["seed"] = num_seed
    dados_gerais_df["Treino_ou_teste"] = 0
    dados_gerais_df.loc[x_test.index,"Treino_ou_teste"] = 1 ## TREINO = 0, TESTE = 1
    dados_gerais_df["Ciclo"] = i
    dados_gerais_df["dG_rosetta"] = y

    #drop non variable features
    var_thr = VarianceThreshold(threshold = 0)
    
    ## transformando nomes das colunas em strings
    x_train.columns = x_train.columns.astype(str)
    x_test.columns = x_test.columns.astype(str)
    
    var_thr.fit(x_train)
    
    concol = [column for column in x_train.columns 
          if column not in x_train.columns[var_thr.get_support()]]
    
    ## removendo colunas com variancia 0
    x_train_no_var_null = x_train.drop(concol,axis=1)
    x_test_no_var_null = x_test.drop(concol,axis=1)

    ## dados_descritores_brutos
    X.columns = X.columns.astype(str)
    descr_data_ = X.drop(concol,axis=1)
    descr_data = X 

    #### Get Mean, SD values for Z-score norms
    scaler = StandardScaler()
    scaler.fit(x_train_no_var_null)
    
    x_train_norm = scaler.transform(x_train_no_var_null)
    x_test_norm = scaler.transform(x_test_no_var_null)
    dados_gerais_norm = scaler.transform(descr_data_)
    dados_gerais_norm = pd.DataFrame(dados_gerais_norm)
    
    return x_train_norm,x_test_norm,y_train,y_test,erro_df,descr_data,dados_gerais_df,dados_gerais_norm,scaler,x_train_no_var_null,x_test_no_var_null

def num_k(data,percent=0.1):
    '''Parâmetros:  
        data = dataset de treino
        percent = porcentagem maxima escolhida levando em conta o N amostral. (DEFAULT = 0.1)
        
        Dessa forma o numero de k é selecionado com base no numero de amostras. 
    '''
        
    data = pd.DataFrame(data)
    
    if 0.1*len(data) < len(data.columns):
                
        numk = int(percent*len(data))
        #print(int(0.1*len(teste)))
        
    else:
        #print(len(teste.columns))
        numk = len(data.columns)
    return numk

def select_best_K_(ka,x_train_normalized,x_test_norm,y_train,dados_gerais_norm):
    
    '''Parâmetros:  
        ka = numero de k
        x_train_normalized = dataset de treino
        x_test_norm = dataset de teste
        y_train = variável resposta de treino
        
        Nessa função é selecionado as melhores k features. Essa seleção é geita utilizando informação mútua.
    '''
    x_train_normalized= pd.DataFrame(x_train_normalized)
    x_test_norm= pd.DataFrame(x_test_norm)
    y_train = column_or_1d(y_train, warn=True)

    
    melhores_k = SelectKBest(score_func=mutual_info_regression,k=ka).fit(x_train_normalized,y_train)
    ## seleciona k features
    cl = melhores_k.get_support(1)
    x_train_normalized_newF = x_train_normalized[x_train_normalized.columns[cl]]
    x_test_normalized_newF = x_test_norm[x_train_normalized_newF.columns]
    
    ## dados totais gerados por esta replica
    dados_pred_geral = dados_gerais_norm[x_train_normalized_newF.columns]
    
    return x_train_normalized_newF,x_test_normalized_newF,cl,dados_pred_geral

## MLR

def lassocv_reg(x_train_normalized_newF, x_test_normalized_newF, y_train, y_test):
    
    folds = 5
    repeats = 5
    
    x_train_norm = pd.DataFrame(x_train_normalized_newF)
    x_test_norm = pd.DataFrame(x_test_normalized_newF)
    y_train = column_or_1d(y_train, warn=True)
    y_test = column_or_1d(y_test, warn=True)
    
    rkf_grid = list(
    RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=42).split(
        x_train_normalized_newF, y_train))

    ## criando o modelo
    lasso_cv = LassoCV(precompute="auto",
    fit_intercept=True,
    max_iter=1000,
    verbose=False,
    eps=1e-04,
    cv=rkf_grid,
    n_alphas=1000,
    n_jobs=10)
    
    # Ajustando o modelo aos dados de treinamento
    lasso_cv.fit(x_train_norm, y_train)

    # Informações
    best_alpha = lasso_cv.alpha_ ## melhor valor de alpha
    coefficients = lasso_cv.coef_  # Coeficientes do modelo após o ajuste

    # Predição
    y_pred = lasso_cv.predict(x_test_norm)
    # Avalie o desempenho do modelo (por exemplo, calculando o erro quadrático médio)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse, coefficients,best_alpha,y_pred,lasso_cv

## RF

def RF_model(x_train_normalized_newF,x_test_normalized_newF,y_train,y_test):
    

# Se y é uma matriz de coluna, você pode usar ravel() para transformá-la em uma matriz unidimensional

    x_train_norm = pd.DataFrame(x_train_normalized_newF)
    x_test_norm = pd.DataFrame(x_test_normalized_newF)
    
    y_train = column_or_1d(y_train, warn=True)
    y_test = column_or_1d(y_test, warn=True)
    
    n_estimators = [10, 20, 50, 100, 500, 1000, 2000]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
    max_depth.append(None)
    min_samples_split = [2, 5, 12,20]
    min_samples_leaf = [1, 2, 4, 8]
    bootstrap = [True, False]

    ### Modelo Floresta Aleatoria
    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap}
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = grid, n_iter = 1000, cv = 3, verbose=2, random_state=42, n_jobs = -1) # Fit the random search model
    
    rf_random.fit(x_train_norm, y_train)
    
    best_params = rf_random.best_params_
    best_rf_model = rf_random.best_estimator_
    
    y_pred = best_rf_model.predict(x_test_norm)
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
    return best_params,best_rf_model,y_pred,rootMeanSqErr

def modelo(dados_desc,i,replica,model):

    x_train_norm,x_test_norm,y_train,y_test,erro_df,descr_data,dados_gerais_df,dados_gerais_norm,scaler,x_train_no_var_null,x_test_no_var_null = pre_process(dados_desc,i)
    
    ka = num_k(x_train_norm,0.1)

    x_train_normalized_newF,x_test_normalized_newF,cl,dados_pred_geral = select_best_K_(ka,x_train_norm,x_test_norm,y_train,dados_gerais_norm)

    if model == 'REG':
        mse, coefficients,best_alpha,y_pred,lasso_cv = lassocv_reg(x_train_normalized_newF,x_test_normalized_newF, y_train, y_test)
        
        # prediçao de dados gerados pelos ciclos dessa replica
        pred_all = lasso_cv.predict(dados_pred_geral)
        
        logging.debug('Salvando arquivos em uma lista')
        list_data_all = [x_train_normalized_newF,y_train, x_test_normalized_newF, y_test,scaler]
        list_data_model = [best_alpha,mse,coefficients,lasso_cv,cl]
        
        logging.debug('Criando arquivos dos datasets')
        # Criar um nome de arquivo único para cada ciclo
        nome_arquivo = f'dados_gerados/dados_dataset_treino_teste_ciclo_REG_{i}_replica_{replica}.pkl'
        
        # Criar um nome de arquivo único para cada ciclo
        with open(nome_arquivo, 'wb') as arquivo:
            pickle.dump(list_data_all, arquivo)
        
        logging.debug('Criando arquivos de infos do modelo')
        # Criar um nome de arquivo único para cada ciclo
        nome_arquivo_1 = f'dados_gerados/dados_infos_modelo_ciclo_{i}_replica_{replica}_REG_.pkl'
        
        # Criar um nome de arquivo único para cada ciclo
        with open(nome_arquivo_1, 'wb') as arquivos:
            pickle.dump(list_data_model, arquivos)
        
    else:
        best_params,best_rf_model,y_pred,mse = RF_model(x_train_normalized_newF,x_test_normalized_newF,y_train,y_test)
        
        # prediçao de dados gerados pelos ciclos dessa replica
        pred_all = best_rf_model.predict(dados_pred_geral)

        list_data_all = [x_train_normalized_newF,y_train, x_test_normalized_newF, y_test,scaler]
        list_data_model = [best_params,best_rf_model,mse,cl]
        
        logging.debug('Criando arquivos dos datasets')
        # Criar um nome de arquivo único para cada ciclo
        nome_arquivo = f'dados_gerados/dados_dataset_treino_teste_ciclo_RF_{i}_replica_{replica}.pkl'
        
        # Criar um nome de arquivo único para cada ciclo
        with open(nome_arquivo, 'wb') as arquivo:
            pickle.dump(list_data_all, arquivo)
        
        logging.debug('Criando arquivos de infos do modelo')
        # Criar um nome de arquivo único para cada ciclo
        nome_arquivo_1 = f'dados_gerados/dados_infos_modelo_ciclo_{i}_replica_{replica}_RF_.pkl'
        
        with open(nome_arquivo_1, 'wb') as arquivos:
            pickle.dump(list_data_model, arquivos)

    ## Dados gerais para predição

    dados_pred_all = pd.DataFrame(pred_all)

    dados_gerais_df["dG_predito"] = dados_pred_all.iloc[:,0]
    erro_modelo = dados_gerais_df.iloc[:,-2] - dados_gerais_df.iloc[:,-1]
    dados_gerais_df["erro_modelo"] = erro_modelo

    ## adicionando valores dos descritores:
    dados_gerais_por_ciclo_df = pd.concat([dados_gerais_df,descr_data],axis=1)

    y_Test_copia = y_test.copy()
    y_Test_copia.reset_index(drop=True, inplace=True)
    dados_erro = pd.DataFrame(y_pred)
    dados_erro["test"] =  y_Test_copia
    dados_erro["erro"] = dados_erro.iloc[:,0] - dados_erro.iloc[:,1]

    return dados_gerais_por_ciclo_df,mse

 
