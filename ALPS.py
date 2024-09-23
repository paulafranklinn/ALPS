#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:27:13 2024

@author: paulafranklinn
"""
import subprocess
import os
import pandas as pd
import random
import numpy as np
from random import choices,choice
from tqdm import tqdm
import time
import logging
import argparse
import pickle
# Importando funções do Rosetta
from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation

# Inicializando o PyRosetta
pyrosetta.init()

# Importando funções personalizadas
from feature_extraction import get_descriptors
from Modelo_MLR_peptide import modelo,preparando_dados_nova_predicao
from Modeling_functions_pyrosetta import Get_residues_from_pose,Get_residues_from_chain
from index_PDB_to_pyRosetta import resi_index_PDB_to_rosetta
from check_and_retrieve import batchs_to_run
from gerar_mutacoes_contri_energy import normalize_function_dG,preparar_para_mutar
from funcoes_gerais_para_organizar_essa_merda import process_data,sortear_sequencias,sele_N_mut,num_total_resid,stats_sequence_que_modelo_direcionou,stats_modelo_explor
from algumacoisa import loop_esmfold_aprofundamento,loop_esmfold_diversidade,chamar_esmfold

# Configuração básica do logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def ALPS(structure, n_loop, seq_numb,n_seq_test,replica,list_residue, list_chain, cpu, model,method,descriptor):

    start_time = time.time()  # Registro do tempo de início
    logging.debug('Starting the Imunobologyc machine')
    
    # Criando pastas para réplicas
    current_directory = os.getcwd()
    dir_name = 'dados_gerados'
    dir_name_pdb = 'PDBs'

    # Verifica se o diretório existe, se não, cria
    if not os.path.exists(os.path.join(current_directory, dir_name)):
        os.mkdir('dados_gerados')
    if not os.path.exists(os.path.join(current_directory, dir_name_pdb)):
        os.mkdir('PDBs')
    
    # Inicialização de variáveis necessárias
    df_dados_gerais = pd.DataFrame()
    DF_seq = pd.DataFrame()
    DF_desc = pd.DataFrame()
    DG_df = pd.DataFrame()
    pdb_struct_list = []
    data_frame_save = pd.DataFrame()
    dados_analise = pd.DataFrame()
    dG_sort_100 = pd.DataFrame()
    dados_analise_ = pd.DataFrame()
    dados_mod_concat = pd.DataFrame()

    scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
    # PDB to pose
    pose = pose_from_pdb(structure)
    
    # Conversão de índice PDB para Rosetta
    index_rosetta_ = resi_index_PDB_to_rosetta(pose, list_chain, list_residue)
    
    # Subtrai 1 de cada índice de resíduo para obter a posição correta
    index_rosetta = index_rosetta_ - 1
    index_rosetta = list(index_rosetta.iloc[:,0])
    
    # Salvar a lista em um arquivo .pkl
    with open('dados_gerados/index_rosetta.pkl', 'wb') as f:
        pickle.dump(index_rosetta, f)
    
    # Salvando a sequência original
    sequencia_original, indice_peptideo = Get_residues_from_pose(pose)
    
    # Contagem do total de resíduos para mutação
    num_residuos_totais = num_total_resid(list_chain,list_residue)
    
    # Sequencia da estrutura inicial
    sequence_to_mute = pose.sequence()

    ## DIVERSIDADE (TRANSFORMAR EM UMA FUNÇÃO) **********
    
    numb_seq_pdb = seq_numb + n_seq_test
    
    seq_mut_df = chamar_esmfold(numb_seq_pdb,sequence_to_mute,num_residuos_totais,'Inicial',1.2)

    logging.debug('Convertendo em descritores')
    lista_sequence_to_modeling = list(seq_mut_df.iloc[:,0])
    data_descriptors = get_descriptors(lista_sequence_to_modeling,descriptor)
    data_d = pd.DataFrame(data_descriptors)

    for i in tqdm(range(n_loop), desc="Processing loops"):

        logging.info(f'Iteração {i+1}/{n_loop}')
        logging.debug(f'Mutações do ciclo {i}')

        for numb_pdb in range(numb_seq_pdb):
            pdb_replic = f'{i}_{numb_pdb}'
            pdb_struct_list.append(pdb_replic)
         
        pdb_listaa = pd.DataFrame(pdb_struct_list)
        #pdb_listaa.to_csv('pdb_listaa.csv')
        
        sequence_cycle, dG_cycle = batchs_to_run(pose, lista_sequence_to_modeling, cpu, i)
                        
        dg_ini = pd.DataFrame(dG_cycle)
        
        DG_df = pd.concat([DG_df, dg_ini],ignore_index=True)
        #DG_df.to_csv('DG_df.csv')

        DF_desc = pd.concat([DF_desc, data_d],ignore_index=True)
        #DF_desc.to_csv('DF_desc.csv')
        
        seque_df = pd.DataFrame(seq_mut_df.iloc[:,0])
        DF_seq = pd.concat([DF_seq, seque_df],ignore_index=True)
        #DF_seq.to_csv('DF_seq.csv')
        
        DF_ = pd.DataFrame()
        DF_ = DF_desc
        DF_['dG'] = DG_df
        DF_['pdb_code'] = pdb_struct_list
        DF_['seq'] = DF_seq
            
        dados_mod_concat = pd.concat([dados_mod_concat,DF_],ignore_index=True)

        #if i > 0:
        #    data_frame_save,dados_analise_ = stats_sequence_que_modelo_direcionou(dG_sort_100,DF_desc,data_frame_save,i,replica,dados_analise_)
        
        Dados_modelo = pd.DataFrame(dados_mod_concat.reset_index(drop=True))        
        #Dados_modelo.to_csv('dados_pre_modelo.csv')

        logging.debug(f'Executando o modelo de {modelo}')
        ## MODELO
        dados_gerais_por_ciclo_df,mse,best_model,concol,scaler,sele_features = modelo(Dados_modelo,i,replica,model,descriptor,n_seq_test)
        
        list_data_model_ = [best_model,concol,scaler,sele_features]

        nome_arquivo = f'dados_nova_pred_{i}_replica_{replica}.pkl'
        with open(nome_arquivo, 'wb') as arquivo:
            pickle.dump(list_data_model_, arquivo)        
        
        ## FAZER
        df_dados_gerais = pd.concat([df_dados_gerais, dados_gerais_por_ciclo_df])
        all_data = pd.DataFrame(df_dados_gerais.reset_index(drop=True))
        all_data.to_csv('dados_gerados/df_dados_gerais_preliminares.csv')
        
        #FAZER FUNÇÃO PARA ANALISE ESTATISTICA
        dados_analise = stats_modelo_explor(all_data,replica,dados_analise,i)
        
        if i != n_loop - 1:
            
            dados_teste = all_data.iloc[:,:7] 

            ## Gerando dados para sorteio 
            df_novas_muts,delta_g_min,delta_g_max = process_data(dados_teste,i)
            #df_novas_muts.to_csv('dg_direct.csv')
            
            # Normalizando dados para sorteio
            dados_norm_dG = normalize_function_dG(df_novas_muts,'dG_predito')
            #dados_norm_dG.to_csv('dados_norm_dG.csv')
            
            # Sorteio da sequência
            dados_norm_dG = sortear_sequencias(dados_norm_dG,10,'pdb_code')
            #dados_norm_dG.to_csv('dados_norm_dG.csv')
 
            # Seleção do número de mutações
            dG_sort_com_n_mut = sele_N_mut(dados_norm_dG,delta_g_min,delta_g_max,num_residuos_totais)
            #dG_sort_com_n_mut.to_csv('dG_sort_com_n_mut.csv')

            ## GERAR 500 SEQUENCIAS A PARTIR DAS 10 SELECIONADAS (50 A PARTIR DE CADA UMA) (TEMP 1)
            
            ## DIVERSIDADE (TRANSFORMAR EM UMA FUNÇÃO)*
            dados_ciclo_aprofundamento = loop_esmfold_aprofundamento(500,dG_sort_com_n_mut,index_rosetta_)
            #dados_ciclo_aprofundamento.to_csv('dados_ciclo_aprofundamento.csv')

            ## SELECIONAR 100 MELHORES USANDO A PREDIÇÃO DO MODELO
            dados_seq_500 = list(dados_ciclo_aprofundamento.iloc[:,0])
            data_descriptors_500 = get_descriptors(dados_seq_500,descriptor)
            data_descriptors_500 = pd.DataFrame(data_descriptors_500)
            
            #data_descriptors_500.to_csv('data_descriptors_500.csv')
            data_prediction_norm_complete = preparando_dados_nova_predicao(data_descriptors_500,concol,scaler,sele_features)
            #data_prediction_norm_complete.to_csv('data_prediction_norm_complete.csv')
            
            df_pred_model = pd.DataFrame()
            df_pred_model['seq'] = pd.DataFrame(dados_ciclo_aprofundamento)
            df_pred_model['dG_predito'] = best_model.predict(data_prediction_norm_complete)
            #df_pred_model.to_csv('df_pred_model_.csv')

            df_pred_model = normalize_function_dG(df_pred_model,'dG_predito')
            
            #df_pred_model.to_csv('df_pred_modelo.csv')

            dG_sort_100 = sortear_sequencias(df_pred_model,100,'seq')

            #dG_sort_100.to_csv('dG_sort_100.csv')
            
            ## GERAR MAIS 200 A PARTIR DAS 100 MELHORES(TEMP 1.2)
            dados_ciclo_diversidade = loop_esmfold_diversidade(200,dG_sort_100,num_residuos_totais)                                         
            #dados_ciclo_diversidade.to_csv('dados_ciclo_diversidade.csv')

            df_aprofund = pd.DataFrame()
            df_aprofund['0'] = dG_sort_100['seq']
            #df_aprofund.to_csv('df_aprofund.csv')    

            df_diversi = pd.DataFrame()
            df_diversi['0'] = dados_ciclo_diversidade
            #df_diversi.to_csv('df_diversi.csv')
            
            seq_mut_df = pd.concat([df_aprofund,df_diversi],ignore_index=True)
            #seq_mut_df.to_csv('seq_mut_df.csv')

            lista_sequence_to_modeling = list(seq_mut_df.iloc[:,0])

            data_descriptors = get_descriptors(lista_sequence_to_modeling,descriptor)
            data_d = pd.DataFrame(data_descriptors)


    # Finaliza a função e grava os resultados
    end_time = time.time()  # Registro do tempo de fim
    execution_time = end_time - start_time  # Cálculo do tempo de execução
    
    logging.debug('Criando CSV com infos gerais')
    all_data.to_csv(f'dados_gerados/df_dados_gerais_replica_{replica}.csv')
    
    logging.debug('Finalizando a função ALPS')
    
    # Escreve o tempo de execução em um arquivo de texto
    with open("exec_time.txt", "w") as f:
        f.write(f"Execution Time: {round(execution_time/60, 1)} minutes\n")

ALPS(structure='2lzt_sem_hetatm_relax_3_times.pdb',
            n_loop=1000,
            seq_numb=200,
            n_seq_test=100,
            replica='L0_1012',
            list_residue=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 65, 66, 67, 68, 69, 70, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 99, 100, 101, 102, 104, 105, 106, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129], 
            list_chain=['A'],
            cpu=40,
            model='RF',
            method='dG',
            descriptor='vhse')
