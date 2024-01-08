
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

import time
import logging

#### Importing rosetta functions
from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation


pyrosetta.init()

## Funções

from gerador_de_seq import gerar_mutacoes
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

def vamo_rodar_poar(sequencia_original,estrutura_pdb,pose,n_loop):
    start_time = time.time()  # Record the start time
    logging.debug('Iniciando a função vamo_rodar_poar')

    mse_list = []
    seq_gerais = []

    logging.info('Gerando mutações iniciais')
    seq_mut_df, posicao_mutacao, posi_mutat_todas,seq_gerais  = gerar_mutacoes(sequencia_original,seq_gerais,50,8)

    for i in range(n_loop):
        logging.info(f'Iteração {i+1}/{n_loop}')

        logging.debug('Executando a função Execute')
        dado_df = Execute(estrutura_pdb, seq_mut_df, "P")

        data_list = list(seq_mut_df[0])
        data_descriptors = get_descriptors(data_list,'vhse')

        data_dff = pd.DataFrame(data_descriptors)
        dG = dado_df[2]
        dG = dG.reset_index()
        data_dff['dG'] = dG[2]
        data_dff['seq'] = seq_mut_df[0]

        logging.debug('Executando o modelo de regressão')
        dados_erro, x_test_norm, best_alpha,erro_df,mse = modelo_regressao(data_dff)
                
        mse_list.append(mse)
                
        dados_seq = data_dff.loc[erro_df.index]
        data_df_erro_new_Seq = dados_erro.iloc[:,1:]
        dados_seq_ = dados_seq.reset_index()
        data_df_erro_new_Seq['test'] = dados_seq_['seq']


        seq_ini = choices(data_df_erro_new_Seq['test'],abs(data_df_erro_new_Seq['erro']), k=1)
        sequencia_original = seq_ini[0]

        logging.debug(f'Sequencia sorteada: {sequencia_original}')

        seq_mut_df,posicao_mutacao,posi_mutat_todas,seq_gerais  = gerar_mutacoes(sequencia_original,seq_gerais,50,3)

    #### Colocar no final da função, antes do return
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    
    # Write the results to a text file
    with open("exec_time.txt", "w") as f:
        f.write(f"Execution Time: {round(execution_time/60, 1)} minutes\n")
    
    logging.info('Finalizando a função vamo_rodar_poar')
    
    return mse_list

exemplo = vamo_rodar_poar(sequencia_original,estrutura_pdb,pose,3)

