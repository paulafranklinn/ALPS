#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:59:01 2024

@author: paula
"""
import pandas as pd
import numpy as np
from sklearn import metrics 

def num_total_resid(list_chain,list_residue):
    
    if len(list_chain) > 1:  # Conta todos os resíduos em todas as cadeias
        total_resi = sum(len(res) for res in list_residue)
        num_residuos_totais = total_resi
    else:
        num_residuos_totais = len(list_residue)
    
    return num_residuos_totais
    
def stats_modelo_ML(data,replica,dados_analise,ciclo):
    
    dados_cic_dG = data.loc[(data["Ciclo"] == ciclo)]

    dG_media = dados_cic_dG['dG'].mean()
    dg_min = dados_cic_dG['dG'].min()
    
    dados_cic_erro = data.loc[(data["Ciclo"] == ciclo)&(data["Treino_ou_teste"] == 1)]
    print(dados_cic_erro)
    r = np.corrcoef(dados_cic_erro["dG"], dados_cic_erro['dG_predito'])[0][1]
    media_erro = abs(dados_cic_erro['erro_modelo']).mean()
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(dados_cic_erro["dG"], dados_cic_erro['dG_predito']))
    dp_erro = np.std(dados_cic_erro['erro_modelo'])
    
    dados_concat = pd.DataFrame({
    'media_dG': [dG_media],
    'min_dG': [dg_min],
    'r': [r],
    'media_erro': [media_erro],
    'RMSE':[rootMeanSqErr],
    'dp':[dp_erro]})

    dados_analise = pd.concat([dados_analise, dados_concat], ignore_index=True)
    
    dados_analise.to_csv(f'dados_gerados/validacao_modelo_r_{replica}.csv')
    
    return dados_analise

def stats_modelo_explor(data,replica,dados_analise,ciclo):
    
    dados_cic_dG = data.loc[(data["Ciclo"] == ciclo)]

    dG_media = dados_cic_dG['dG'].mean()
    dg_min = dados_cic_dG['dG'].min()
    
    r = np.corrcoef(dados_cic_dG["dG"], dados_cic_dG['dG_predito'])[0][1]
    media_erro = abs(dados_cic_dG['erro_modelo']).mean()
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(dados_cic_dG["dG"], dados_cic_dG['dG_predito']))

    dados_concat = pd.DataFrame({
    'media_dG': [dG_media],
    'min_dG': [dg_min],
    'r': [r],
    'media_erro': [media_erro],
    'RMSE': [rootMeanSqErr]})

    dados_analise = pd.concat([dados_analise, dados_concat], ignore_index=True)
    
    dados_analise.to_csv(f'dados_gerados/analise_direcionamento_modelo_r_{replica}.csv')
    
    return dados_analise

def stats_sequence_que_modelo_direcionou(dG_sort_100,DF_desc,data_frame_save,ciclo,replica,data_frame_stats_direcionamento):
        
    seq = pd.DataFrame()
    seq['seq'] = dG_sort_100['seq']
    
    df_merged = pd.merge(seq['seq'], DF_desc, on='seq', how='inner')

    df_merged['dG_pred'] = dG_sort_100['dG_predito']
    df_merged['Ciclo'] = ciclo
    
    data_frame_save = pd.concat([data_frame_save,df_merged],ignore_index=True)
    
    data_frame_save.to_csv('dados_gerados/dados_direcionamento_r_{replica}.csv')
    
    dados_analise_ = stats_modelo_explor(data_frame_save,replica,data_frame_stats_direcionamento,ciclo)
    
    return data_frame_save,dados_analise_

def process_data(all_data, ciclo):
    # Filtra os dados pelo ciclo especificado
    df_novas_muts = all_data[all_data["Ciclo"] == ciclo]
    
    delta_g_min = min(df_novas_muts['dG_predito'])
    delta_g_max = max(df_novas_muts['dG_predito'])
    
    return df_novas_muts,delta_g_min,delta_g_max

def sortear_sequencias(dg_direct,k,Columna):
    
    # Sorteio de 'k' valores únicos da coluna "pdb_code", respeitando o peso de "dG_predito_normalized_invert"
    sequencias_sorteadas = np.random.choice(
        dg_direct[f"{Columna}"],
        size=k,
        replace=False,  # Garante que não haja repetição
        p=dg_direct["dG_predito_normalized_invert"]/dg_direct["dG_predito_normalized_invert"].sum())
    
    dG_sort = dg_direct[dg_direct[f"{Columna}"].isin(sequencias_sorteadas)]            

    return dG_sort

def sele_N_mut(dG_sort,delta_g_min,delta_g_max,num_residuos_totais):
    
    N_mut_list = []
    
    N_mut_min = 0.08*num_residuos_totais
    N_mut_max = num_residuos_totais
    delta_g_range = delta_g_max - delta_g_min
    
    dG_sort = dG_sort.reset_index(drop=True)

    for i in dG_sort['dG_predito']:
        dg = i
        N_mut = N_mut_min + ((dg - delta_g_min) / delta_g_range) * (N_mut_max - N_mut_min)
        N_mut_rounded = round(N_mut)
        
        N_mut_list.append(N_mut_rounded)
        
    df_N_mut_list = pd.DataFrame()
    df_N_mut_list['sera_que_foi'] = N_mut_list
    df_N_mut_list.to_csv('df_N_mut_list.csv')
    
    dG_sort['N_mut'] = N_mut_list

    return dG_sort
        
