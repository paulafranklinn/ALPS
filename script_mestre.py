
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

#### Importing rosetta functions
from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation



pyrosetta.init()

## Functions
from gerador_de_seq import gerar_mutacoes
from descriptor import assign_descriptor,get_descriptors
from modeling_functions_pyrosetta import mutate_repack, model_sequence, Execute, Get_residues_from_chain
from Modelo_MLR_peptide import pre_process,num_k,select_best_K_,lassocv_reg,modelo_regressao


## arquivos importantes
estrutura_pdb = "3mre_recortada_relax.pdb"
pose = pose_from_pdb("3mre_recortada_relax.pdb")
scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
scorefxn(pose)
sequencia_original, indice_peptideo = Get_residues_from_chain(pose,"P")


def vamo_rodar_poar(sequencia_original,estrutura_pdb,pose):

    mse_list = []
    seq_gerais = []

    for i in range(3):
        seq_mut_df, posicao_mutacao, posi_mutat_todas,seq_gerais  = gerar_mutacoes(sequencia_original,seq_gerais,50,8)

        dado_df = Execute(estrutura_pdb, seq_mut_df, "P")

        data_list = list(seq_mut_df[0])
        data_descriptors = get_descriptors(data_list,'vhse')

        data_dff = pd.DataFrame(data_descriptors)
        dG = dado_df[2]
        dG = dG.reset_index()
        data_dff['dG'] = dG[2]
        data_dff['seq'] = seq_mut_df[0]
        dados_erro, x_test_norm, best_alpha,erro_df,mse = modelo_regressao(data_dff)
                
        mse_list.append(mse)
                
        dados_seq = data_dff.loc[erro_df.index]
        data_df_erro_new_Seq = dados_erro.iloc[:,1:]
        dados_seq_ = dados_seq.reset_index()
        data_df_erro_new_Seq['test'] = dados_seq_['seq']


        seq_ini = choices(data_df_erro_new_Seq['test'],abs(data_df_erro_new_Seq['erro']), k=1)
        sequencia_original = seq_ini[0]
        print(sequencia_original)

    return mse_list





