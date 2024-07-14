#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:27:13 2024

@author: papaula
"""

import os
import pandas as pd
from random import choices
from tqdm import tqdm
import time
import logging
import argparse

# Importando funções do Rosetta
from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation

# Inicializando o PyRosetta
pyrosetta.init()

# Importando funções personalizadas
from gerador_de_sequencia import gerar_mutacoes
from feature_extraction import get_descriptors
from Modelo_MLR_peptide import modelo
from Modeling_functions_pyrosetta import Get_residues_from_pose
from index_PDB_to_pyRosetta import resi_index_PDB_to_rosetta
from check_and_retrieve import batchs_to_run

# Configuração básica do logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def peptide_machine(pose, n_loop, seq_numb, replica, list_residue, list_chain, chain, cpu, model):

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
    mse_list = []
    df_dados_gerais = pd.DataFrame()
    DF_seq = pd.DataFrame()
    DF_desc = pd.DataFrame()
    DG_df = pd.DataFrame()
    sequencias_start = []
    list_seq = []
    
    # Conversão de índice PDB para Rosetta
    index_rosetta = resi_index_PDB_to_rosetta(pose, list_chain, list_residue)
    
    # Salvando a sequência original
    sequencia_original, indice_peptideo = Get_residues_from_pose(pose)
    sequencias_start.append(sequencia_original)
    
    # Contagem do total de resíduos para mutação
    if len(list_chain) > 1:  # Conta todos os resíduos em todas as cadeias
        total_resi = sum(len(res) for res in list_residue)
        num_residuos_totais = total_resi
    else:
        num_residuos_totais = len(list_residue)
    
    # Subtrai 1 de cada índice de resíduo para obter a posição correta
    index_rosetta = index_rosetta - 1
    list_para_mutar = list(index_rosetta.iloc[:,0])
    
    # Gerando mutações iniciais
    logging.info('Gerando mutações iniciais')
    seq_mut_df, list_seq = gerar_mutacoes(sequencia_original, list_seq, seq_numb, num_residuos_totais, list_para_mutar)  

    for i in tqdm(range(n_loop), desc="Processing loops"):
        
        logging.info(f'Iteração {i+1}/{n_loop}')
        logging.debug('Modeling and calculating dG')
        
        lista_sequence_to_modeling = list(seq_mut_df.iloc[:,0])
        sequence_cycle, dG_cycle = batchs_to_run(pose, lista_sequence_to_modeling, cpu, chain, i)
        
        dg_ini = pd.DataFrame(dG_cycle)
        DG_df = pd.concat([DG_df, dg_ini])
        
        logging.debug('Convertendo em descritores')
        dados_seq = list(seq_mut_df[0])
        data_descriptors = get_descriptors(dados_seq, 'vhse')
        data_d = pd.DataFrame(data_descriptors)
        DF_desc = pd.concat([DF_desc, data_d])
        
        seque_df = pd.DataFrame(seq_mut_df)
        DF_seq = pd.concat([DF_seq, seque_df])
        
        # Adequando outputs do get_descriptors para a função modelo_regressão
        DF_desc['dG'] = DG_df
        DF_desc['seq'] = DF_seq
        
        DFF = pd.DataFrame(DF_desc.reset_index())
        Dados_modelo = DFF.iloc[:,1:]
        
        logging.debug(f'Executando o modelo de {modelo}')
        dados_gerais_por_ciclo_df, mse = modelo(Dados_modelo, i, replica, model)
                
        df_dados_gerais = pd.concat([df_dados_gerais, dados_gerais_por_ciclo_df])
        df_dados_gerais_ = pd.DataFrame(df_dados_gerais.reset_index())
        all_data = df_dados_gerais_.iloc[:,1:]
        all_data.to_csv('dados_gerados/df_dados_gerais_preliminares.csv')
        
        mse_list.append(mse)
        
        if i != n_loop - 1:
            df_novas_muts = all_data[all_data["Ciclo"] == i].reset_index()
            df_novas_muts_ = df_novas_muts.iloc[:,1:]
            df_dados_erro_sorteio = pd.DataFrame()
            
            df_dados_erro_sorteio["seq"] = df_novas_muts_["seq"]
            df_dados_erro_sorteio["erro_modelo"] = df_novas_muts_["erro_modelo"].abs()
            
            # Sorteio da sequência
            seq_ini = choices(df_dados_erro_sorteio["seq"], df_dados_erro_sorteio["erro_modelo"], k=1)
            
            logging.debug(f'Sequencia sorteada: {sequencia_original}')
            
            sequencia_original = seq_ini[0]
            sequencias_start.append(sequencia_original)
            
            # Seleção do número de mutações
            erro_sort = df_dados_erro_sorteio[df_dados_erro_sorteio["seq"] == sequencia_original]
            Err_sorteado = erro_sort.iloc[0,1]
            erro_max = max(df_dados_erro_sorteio["erro_modelo"])
            Proporcao = (num_residuos_totais / erro_max)
            X_num = round(Err_sorteado * Proporcao)
            N_mut_ = num_residuos_totais - X_num
            
            if N_mut_ < 1:
                N_mut = 1
            else:
                N_mut = N_mut_
            
            seq_mut_df, list_seq = gerar_mutacoes(sequencia_original, list_seq, seq_numb, N_mut, list_para_mutar)  
    
    # Finaliza a função e grava os resultados
    end_time = time.time()  # Registro do tempo de fim
    execution_time = end_time - start_time  # Cálculo do tempo de execução
    
    df_seq_originais = pd.DataFrame(sequencias_start)
    df_seq_originais.to_csv(f'dados_gerados/df_seq_originais_replica_{replica}.csv')
    
    logging.debug('Criando CSV com infos gerais')
    all_data.to_csv(f'dados_gerados/df_dados_gerais_replica_{replica}.csv')
    
    logging.debug('Finalizando a função vamo_rodar_poar')
    
    # Escreve o tempo de execução em um arquivo de texto
    with open("exec_time.txt", "w") as f:
        f.write(f"Execution Time: {round(execution_time/60, 1)} minutes\n")
    
    return mse_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run peptide machine simulation.')
    parser.add_argument('--estrutura_pdb', type=str, required=True, help='Path to the PDB structure file.')
    parser.add_argument('--n_loop', type=int, required=True, help='Number of loops.')
    parser.add_argument('--seq_numb', type=int, required=True, help='Number of sequences.')
    parser.add_argument('--replica', type=int, required=True, help='Replica number.')
    parser.add_argument('--list_residue', type=int, nargs='+', required=True, help='List of residues.')
    parser.add_argument('--list_chain', type=str, nargs='+', required=True, help='List of chains.')
    parser.add_argument('--chain', type=str, required=True, help='Chain identifier.')
    parser.add_argument('--cpu', type=int, required=True, help='Number of CPUs to use.')
    parser.add_argument('--model', type=str, required=True, help='Model to use.')

    args = parser.parse_args()

    pose = pose_from_pdb(args.estrutura_pdb)
    
    teste = peptide_machine(pose=pose,
                            n_loop=args.n_loop,
                            seq_numb=args.seq_numb,
                            replica=args.replica,
                            list_residue=args.list_residue,
                            list_chain=args.list_chain,
                            chain=args.chain,
                            cpu=args.cpu,
                            model=args.model)
