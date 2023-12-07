# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""

import random
import csv
import pandas as pd
import numpy as np

def gerar_mutacoes(sequencia_original, num_mutacoes):
    sequencias_mutadas = []
    seq_mut_df = 0
    
    while len(sequencias_mutadas) < num_mutacoes:
        posicao_mutacao = random.randint(0, len(sequencia_original) - 1)
        novo_aminoacido = random.choice("ACDEFGHIKLMNPQRSTVWY")
    
        sequencia_mutada = list(sequencia_original)
        sequencia_mutada[posicao_mutacao] = novo_aminoacido
        sequencia_mutada = ''.join(sequencia_mutada)
            
        if sequencia_mutada not in sequencias_mutadas:            
            sequencias_mutadas.append(sequencia_mutada)
        
        porcentagem_concluida = (len(sequencias_mutadas) / num_mutacoes) * 100
        print(f"({porcentagem_concluida:.2f}% concluído)")

    seq_mut_df = pd.DataFrame(sequencias_mutadas)

    return seq_mut_df


## Como usar?
sequencia_original = "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTAAAA" ## especifica sequencia

b = gerar_mutacoes(sequencia_original, 1000) ## roda funçao
