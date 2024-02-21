#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:07:40 2023

@author: paula
"""

# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""

import random
import csv
import pandas as pd
import numpy as np

def gerar_mutacoes(sequencia_original,dados_seq_anteriores,n_popul,n_mut,list_para_mutar):
    
    ## variáveis
    sequencias_mutadas = []
    seq_mut_df = 0
    posi_mutat_todas= []

    if list_para_mutar != 0:
        posi_para_mut = list_para_mutar
    else:
        posi_para_mut = range(len(sequencia_original))

    while len(sequencias_mutadas) < n_popul:
        posicao_mutacao = random.sample(posi_para_mut,k=n_mut)
        posi_mutat_todas.append(posicao_mutacao)
        novo_aminoacido = random.choices("ACDEFGHIKLMNPQRSTVWY",k=n_mut)


        sequencia_mutada = list(sequencia_original)
        
        for i in range(n_mut):
            if sequencia_mutada[posicao_mutacao[i]] != novo_aminoacido[i]:
                sequencia_mutada[posicao_mutacao[i]] = novo_aminoacido[i]
            
        sequencia_mutada = ''.join(sequencia_mutada)
        
        if sequencia_mutada not in sequencias_mutadas and sequencia_mutada not in dados_seq_anteriores:            
            sequencias_mutadas.append(sequencia_mutada)

        #porcentagem_concluida = (len(sequencias_mutadas) / n_popul) * 100
        #print(f"({porcentagem_concluida:.2f}% concluído)")
    seq_mut_df = pd.DataFrame(sequencias_mutadas)
    ##return seq_mut_df, posicao_mutacao, posi_mutat_todas
    return seq_mut_df

## Como usar?
#sequencia_original = "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTAAAA" ## especifica sequencia
#testando = gerar_mutacoes(sequencia_original,10,4) ## roda funçao
