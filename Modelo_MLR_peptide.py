
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

def pre_process(data):
    dataset = data.drop_duplicates(subset = data.iloc[:,:-1])

    X = dataset.iloc[:,:-2]
    y = dataset['dG']

    # Split the data into training and testing sets with a random split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    erro_df = pd.DataFrame(x_test)
    
    #### Get Mean, SD values for Z-score norms
    scaler = StandardScaler()
    scaler.fit(x_train)
    
    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)
    StandardScaler()
    return x_train_norm, x_test_norm, y_train, y_test, erro_df

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

def select_best_K_(ka,x_train_normalized,x_test_norm,y_train):
    
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

    return x_train_normalized_newF, x_test_normalized_newF

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
    n_jobs=10,)
    
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
    
    return mse, coefficients,best_alpha,y_pred,y_test

def modelo_regressao(dados_desc):

    x_train_norm, x_test_norm, y_train, y_test, erro_df = pre_process(dados_desc)
    
    ka = num_k(x_train_norm,0.1)

    x_train_normalized_newF, x_test_normalized_newF = select_best_K_(ka,x_train_norm,x_test_norm,y_train)

    mse, coefficients,best_alpha,y_pred,y_test = lassocv_reg(x_train_normalized_newF,x_test_normalized_newF, y_train, y_test)
    
    y_Test_copia = y_test.copy()
    y_Test_copia.reset_index(drop=True, inplace=True)
    dados_erro = pd.DataFrame(y_pred)
    dados_erro["test"] =  y_Test_copia
    dados_erro["erro"] = dados_erro.iloc[:,0] - dados_erro.iloc[:,1]

    return dados_erro, x_test_norm, best_alpha,erro_df,mse
