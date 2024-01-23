
import os
os.chdir("/home/papaula/Documents/Projetos/Mestrado/peptide_machine-main/")

## bibliotecas
import os
import numpy as np
from numpy.random import uniform
from random import sample
import pandas as pd
from random import choices
from tqdm import tqdm
from time import sleep
import pickle

import time
import logging

#### Importing rosetta functions
from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation


pyrosetta.init()

## Funções

from gerador_de_sequencia import gerar_mutacoes
from feature_extraction import assign_descriptor,get_descriptors
from modeling_functions_pyrosetta import mutate_repack, model_sequence, Execute, Get_residues_from_chain
from Modelo_MLR_peptide import pre_process,num_k,select_best_K_,lassocv_reg,modelo_regressao

## arquivos importantes
estrutura_pdb = "3mre_recortada_relax.pdb"
pose = pose_from_pdb("3mre_recortada_relax.pdb")
scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
scorefxn(pose)
sequencia_original, indice_peptideo = Get_residues_from_chain(pose,"P")

# Configuração básica do logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def vamo_rodar_poar(sequencia_original,estrutura_pdb,n_loop):
    start_time = time.time()  # Record the start time
    logging.debug('Iniciando a função vamo_rodar_poar')

    mse_list = []
    df_dados_gerais = pd.DataFrame()
    DF_seq = pd.DataFrame()
    DF_desc = pd.DataFrame()
    DG_df = pd.DataFrame()
    list_seq = []
    sequencias_start = []
    
    ## adicionando dados de sequencias usadas para a mutação
    sequencias_start.append(sequencia_original)

    logging.info('Gerando mutações iniciais')
    seq_mut_df = gerar_mutacoes(sequencia_original,list_seq,50,8)
    

    for i in range(n_loop):
        logging.info(f'Iteração {i+1}/{n_loop}')

        logging.debug('Executando a função Execute')
        dado_df = Execute(estrutura_pdb,seq_mut_df, "P")
        dg_ini = pd.DataFrame(dado_df[2]).reset_index()
        DG_df = pd.concat([DG_df,dg_ini[2]])

        logging.debug('Convertendo em descritores')
        dados_seq =list(seq_mut_df[0])
        data_descriptors = get_descriptors(dados_seq,'vhse')
        data_d = pd.DataFrame(data_descriptors)
        DF_desc = pd.concat([DF_desc,data_d])

        seque_df = pd.DataFrame(seq_mut_df)
        DF_seq = pd.concat([DF_seq,seque_df])

        ##Adequanto outputs do get_descriptors para a funçao modelo_regressão
        DF_desc['dG'] = DG_df
        DF_desc['seq'] = DF_seq

        DFF = pd.DataFrame(DF_desc.reset_index())
        Dados_modelo = DFF.iloc[:,1:]
        Dados_modelo.to_csv(f'Dados_modelo{i}.csv')

        logging.debug('Executando o modelo de regressão')
        x_train_normalized_newF,y_train, x_test_normalized_newF, y_test,y_pred,dados_erro,best_alpha,mse,coefficients,erro_df,lasso_cv,cl,dados_gerais_por_ciclo_df = modelo_regressao(Dados_modelo,i)

        df_dados_gerais = pd.concat([df_dados_gerais,dados_gerais_por_ciclo_df])
        df_dados_gerais_ = pd.DataFrame(df_dados_gerais.reset_index())
        all_data = df_dados_gerais_.iloc[:,1:]

        logging.debug('Salvando arquivos em uma lista')
        list_data_all = [x_train_normalized_newF,y_train, x_test_normalized_newF, y_test]
        list_data_model = [best_alpha,mse,coefficients,lasso_cv,cl]

        logging.debug('Criando arquivos dos datasets')
        # Criar um nome de arquivo único para cada ciclo
        nome_arquivo = f'Output/dados_dataset_treino_teste_ciclo_{i}.pkl'

        # Criar um nome de arquivo único para cada ciclo
        with open(nome_arquivo, 'wb') as arquivo:
            pickle.dump(list_data_all, arquivo)

        logging.debug('Criando arquivos de infos do modelo')
        # Criar um nome de arquivo único para cada ciclo
        nome_arquivo_1 = f'Output/dados_infos_modelo_ciclo_{i}.pkl'

        # Criar um nome de arquivo único para cada ciclo
        with open(nome_arquivo_1, 'wb') as arquivos:
            pickle.dump(list_data_model, arquivos)

        mse_list.append(mse)
        
        logging.debug('Testando se o erro é aqui')
        
        if i != n_loop-1:
            sequencia = all_data.iloc[:,0]
            erro = all_data.iloc[:,5]

            seq_ini = choices(sequencia,erro.abs(), k=1)
            sequencia_original = seq_ini[0]
            sequencias_start.append(sequencia_original)

            logging.debug(f'Sequencia sorteada: {sequencia_original}')
            seq_mut_df = gerar_mutacoes(sequencia_original,dados_seq,50,3)

    #### Colocar no final da função, antes do return
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    
    df_seq_originais = pd.DataFrame(sequencias_start)
    df_seq_originais.to_csv('df_seq_originais.csv')                                 

    logging.debug('Criando CSV com infos gerais')
    all_data.to_csv('df_dados_gerais.csv')

    logging.debug('Finalizando a função vamo_rodar_poar')

    # Write the results to a text file
    with open("exec_time.txt", "w") as f:
        f.write(f"Execution Time: {round(execution_time/60, 1)} minutes\n")
            
    return mse_list

exemplo = vamo_rodar_poar(sequencia_original,estrutura_pdb,20)




