
## bibliotecas
#import numpy as np
import pandas as pd

## Pre-processamento dos dados
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,mutual_info_regression

## Algoritmos
from sklearn.linear_model import LassoCV
#from xgboost import XGBRegressor
#from sklearn.neural_network import MLPRegressor

## analise dos modelos
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold

# Load the data from a CSV file
#csv_file = "/home/papaula/Documentos/Mestrado//Peptides_as_descriptors_final_withdGandMethods_v02.csv"
#data = pd.read_csv(csv_file)
#print(data)

def pre_process(data,i):

    num_seed = (42+i)

    dataset = data

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

    ## dados_descritores_brutos
    descr_data_ = X
    descr_data = X
    #### Get Mean, SD values for Z-score norms
    scaler = StandardScaler()
    scaler.fit(x_train)
    
    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)
    dados_gerais_norm = scaler.transform(descr_data_)
    dados_gerais_norm = pd.DataFrame(dados_gerais_norm)
    StandardScaler()

    return x_train_norm, x_test_norm, y_train, y_test, erro_df,descr_data,dados_gerais_df,dados_gerais_norm

def num_k(data,percent=0.1):
    '''Parâmetros:  
        data = dataset de treino
        percent = porcentagem maxima escolhida levando em conta o N amostral. (DEFAULT = 0.1)
        
        Dessa forma o numero de k é selecionado com base no numero de amostras. 
    '''
    data = pd.DataFrame(data)
    if len(data.columns) > percent*len(data):
        numk = int(percent*len(data))
    else:
        numk =len(data.columns)
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
    y_train = pd.DataFrame(y_train)
    
    melhores_k = SelectKBest(score_func=mutual_info_regression,k=ka).fit(x_train_normalized,y_train)
    ## seleciona k features
    cl = melhores_k.get_support(1)
    x_train_normalized_newF = x_train_normalized[x_train_normalized.columns[cl]]
    x_test_normalized_newF = x_test_norm[x_train_normalized_newF.columns]

    dados_pred_geral = dados_gerais_norm[x_train_normalized_newF.columns]

    return x_train_normalized_newF,x_test_normalized_newF,cl,dados_pred_geral

def lassocv_reg(x_train_normalized_newF, x_test_normalized_newF, y_train, y_test):
    
    folds = 10
    repeats = 10
    
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
    
    x_train_norm = pd.DataFrame(x_train_normalized_newF)
    x_test_norm = pd.DataFrame(x_test_normalized_newF)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    # Ajustando o modelo aos dados de treinamento
    lasso_cv.fit(x_train_norm, y_train)

    # Informações
    best_alpha = lasso_cv.alpha_ ## melhor valor de alpha
    coefficients = lasso_cv.coef_  # Coeficientes do modelo após o ajuste

    # Predição
    y_pred = lasso_cv.predict(x_test_norm)
    # Avalie o desempenho do modelo (por exemplo, calculando o erro quadrático médio)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse, coefficients,best_alpha,y_pred,y_test,lasso_cv

def modelo_regressao(dados_desc,i):

    x_train_norm, x_test_norm, y_train, y_test, erro_df,descr_data,dados_gerais_df,dados_gerais_norm = pre_process(dados_desc,i)
    
    ka = num_k(x_train_norm,0.1)

    x_train_normalized_newF, x_test_normalized_newF,cl,dados_pred_geral = select_best_K_(ka,x_train_norm,x_test_norm,y_train,dados_gerais_norm)

    mse, coefficients,best_alpha,y_pred,y_test,lasso_cv = lassocv_reg(x_train_normalized_newF,x_test_normalized_newF, y_train, y_test)

    ## Dados gerais para predição
    pred_all = lasso_cv.predict(dados_pred_geral)

    dados_pred_all = pd.DataFrame(pred_all)

    dados_gerais_df["dG_predito"] = dados_pred_all
    erro_modelo = dados_gerais_df.iloc[:,-2] - dados_gerais_df.iloc[:,-1]
    dados_gerais_df["erro_modelo"] = erro_modelo

    ## adicionando valores dos descritores:
    for add_col_dec in range(len(descr_data.columns)):
        dados_gerais_df[f"Descritor_{add_col_dec}"] = descr_data.iloc[:,add_col_dec]

    dados_gerais_por_ciclo_df = dados_gerais_df

    y_Test_copia = y_test.copy()
    y_Test_copia.reset_index(drop=True, inplace=True)
    dados_erro = pd.DataFrame(y_pred)
    dados_erro["test"] =  y_Test_copia
    dados_erro["erro"] = dados_erro.iloc[:,0] - dados_erro.iloc[:,1]

    return x_train_normalized_newF,y_train, x_test_normalized_newF, y_test,y_pred,dados_erro,best_alpha,mse,coefficients,erro_df,lasso_cv,cl,dados_gerais_por_ciclo_df
