#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 23:21:36 2024

@author: papaula
"""

import torch
import esm
import random
import pandas as pd
import numpy as np
from random import choices
import argparse
import pickle
model, alphabet = esm.pretrained.load_model_and_alphabet_local("/home/paula.franklin/proj.paula.franklin/Mestrado/modelo_esm/esm2_t33_650M_UR50D.pt")
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

def select_and_remove(df, index):
    selected_row = df.loc[index]
    df = df.drop(index)
    return df, selected_row

# Função principal para selecionar e remover linhas ponderadas
def select_weighted_positions(data_df, number_position):
    list_position = []

    while len(list_position) < number_position and not data_df.empty:
        # Seleciona uma linha ponderada pela coluna 'peso'
        data_selected = choices(data_df.index, weights= data_df['cont_energy_normalized'], k=1)[0]
        data_df, selected_row = select_and_remove(data_df, data_selected)
        list_position.append(selected_row['index'])

    return list_position

def insert_mask(sequence, position, mask="<mask>"):
    """
    Replaces a character in a given position of a sequence with a mask.

    Parameters:
    - sequence (str or list): The sequence to replace the character in.
    - position (int): The position in the sequence where the character should be replaced.
    - mask (str): The mask to insert (default is "<mask>").

    Returns:
    - str or list: The sequence with the mask replacing the character at the specified position.
    """

    if not (0 <= position < len(sequence)):
        raise ValueError("Position is out of bounds.")

    if isinstance(sequence, str):
        return sequence[:position] + mask + sequence[position + 1:]
    elif isinstance(sequence, list):
        return sequence[:position] + [mask] + sequence[position + 1:]
    else:
        raise TypeError("Sequence must be a string or list.")

def complete_mask(input_sequence, posi, temperature=1.0):

    standard_aa = [alphabet.get_idx(aa) for aa in ['A', 'R', 'N', 'D', 'C', 'Q', 
                                                   'E', 'G', 'H', 'I', 'L', 'K', 
                                                   'M', 'F', 'P', 'S', 'T', 'W', 
                                                   'Y', 'V']]

    data = [
        ("protein1", insert_mask(input_sequence, posi, mask="<mask>"))]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Predict masked tokens
    with torch.no_grad():
        token_probs = model(batch_tokens, repr_layers=[33])["logits"]

    # Apply temperature
    token_probs /= temperature

    softmax = torch.nn.Softmax(dim=-1)
    probabilities = softmax(token_probs)

    # Get the index of the <mask> token
    mask_idx = (batch_tokens == alphabet.mask_idx).nonzero(as_tuple=True)

        # Zero out probabilities for excluded tokens
        
    for token_idx in range(probabilities.size(-1)):
        if token_idx not in standard_aa:
            probabilities[:, :, token_idx] = 0.0

    # Sample from the probability distribution
    predicted_tokens = torch.multinomial(probabilities[mask_idx], num_samples=1).squeeze(-1)

    # Replace the <mask> token with the predicted token
    batch_tokens[mask_idx] = predicted_tokens

    predicted_residues = [alphabet.get_tok(pred.item()) for pred in batch_tokens[0]]

    seq_predicted = ''.join(predicted_residues[1:-1])

    if input_sequence != seq_predicted:
        print("Mutation added!! 😉")

    return seq_predicted

def create_masked_sequences(sequence, masked_pos, cdrs_or_fm):

    if cdrs_or_fm == "cdr":
        masked_sequences = []
        for i in range(len(sequence)):
            if masked_pos[i] == 1:
                masked_sequence = sequence[:i] + "<mask>" + sequence[i+1:]
                masked_sequences.append((f"protein1 with mask at position {i+1}", masked_sequence))
        return masked_sequences
    if cdrs_or_fm == "fm":
        masked_sequences = []
        for i in range(len(sequence)):
            if masked_pos[i] == 0:
                masked_sequence = sequence[:i] + "<mask>" + sequence[i+1:]
                masked_sequences.append((f"protein1 with mask at position {i+1}", masked_sequence))
        return masked_sequences

def generate_Sequence(input_sequence, cdrs, loc, temperature = 1.0, order='random'):

    new_sequence_temp = input_sequence

    indices = list(range(len(input_sequence)))

    if loc == "all":
        if order == 'random':
            random.shuffle(indices)
        elif order == 'forward':
            pass  # already in forward order
        elif order == 'backward':
            indices.reverse()
        for i in indices:
            print(i)
            new_sequence_temp = complete_mask(input_sequence=new_sequence_temp, posi=i, temperature=temperature)

    elif loc == "cdr":
        cdr_indices = [i for i in indices if i in cdrs]
        if order == 'random':
            random.shuffle(cdr_indices)
        elif order == 'forward':
            pass  # already in forward order
        elif order == 'backward':
            cdr_indices.reverse()
        for i in cdr_indices:
            print(i)
            new_sequence_temp = complete_mask(input_sequence=new_sequence_temp, posi=i, temperature=temperature)

    elif loc == "fm":
        fm_indices = [i for i in indices if i not in cdrs]
        if order == 'random':
            random.shuffle(fm_indices)
        elif order == 'forward':
            pass  # already in forward order
        elif order == 'backward':
            fm_indices.reverse()
        for i in fm_indices:
            new_sequence_temp = complete_mask(input_sequence=new_sequence_temp, posi=i, temperature=temperature)

    return new_sequence_temp

def esmfold_contrib_energy(n_popul,sequence_to_mute,n_mut,initial,temp):
    ## variáveis

    dados_seq_anteriores = []
    sequencias_mutadas = []
    index_rosetta = pd.read_pickle('dados_gerados/index_rosetta.pkl')

    if initial == 'Inicial':

        dados_seq_anteriores = []
        
        while len(sequencias_mutadas) < n_popul:

            sequencia_mutada = generate_Sequence(input_sequence=sequence_to_mute,
                                                                cdrs=index_rosetta,
                                                                temperature=temp,
                                                                loc="cdr",
                                                                order = "random")

            if sequencia_mutada not in sequencias_mutadas and sequencia_mutada not in dados_seq_anteriores:

                sequencias_mutadas.append(sequencia_mutada)
                dados_seq_anteriores.append(sequencia_mutada)

    elif initial == 'Aprofundamento':

        dados_seq_anteriores = pd.read_pickle('dados_gerados/dados_seq_anteriores.pkl')
        posi_to_mute = pd.read_csv('dados_gerados/posi_to_mute.csv')
        index_rosetta = select_weighted_positions(posi_to_mute,n_mut)
        
        while len(sequencias_mutadas) < n_popul:

            sequencia_mutada = generate_Sequence(input_sequence=sequence_to_mute,
                                                                cdrs=index_rosetta,
                                                                temperature=temp,
                                                                loc="cdr",
                                                                order = "random")

            if sequencia_mutada not in sequencias_mutadas and sequencia_mutada not in dados_seq_anteriores:

                sequencias_mutadas.append(sequencia_mutada)
                dados_seq_anteriores.append(sequencia_mutada)
        
    elif initial == 'Diversidade':
        
        dados_seq_anteriores = pd.read_pickle('dados_gerados/dados_seq_anteriores.pkl')        
        index_rosetta = pd.read_pickle('dados_gerados/index_rosetta.pkl')

        while len(sequencias_mutadas) < n_popul:
    
            sequencia_mutada = generate_Sequence(input_sequence=sequence_to_mute,
                                                                cdrs=index_rosetta,
                                                                temperature=temp,
                                                                loc="cdr",
                                                                order = "random")
    
            if sequencia_mutada not in sequencias_mutadas and sequencia_mutada not in dados_seq_anteriores:
    
                sequencias_mutadas.append(sequencia_mutada)
                dados_seq_anteriores.append(sequencia_mutada)

    seq_mut_df = pd.DataFrame(sequencias_mutadas)
    seq_mut_df.to_csv('dados_gerados/seq_mut_df.csv')

    # Salvar a lista em um arquivo .pkl
    with open('dados_gerados/dados_seq_anteriores.pkl', 'wb') as f:
        pickle.dump(dados_seq_anteriores, f)

parser = argparse.ArgumentParser(description="Run the esmfold function with specified parameters.")
parser.add_argument("--n_popul", type=int, required=True, help="Number of mutated sequences to generate.")
parser.add_argument("--sequence_to_mute", type=str, required=True, help="The sequence to be mutated.")
parser.add_argument("--n_mut", type=int, required=True, help="Number of mutations.")
parser.add_argument("--initial", type=str, default='Y', help="Mutation parameter.")
parser.add_argument("--temp", type=float, default=1.0, help="Temperature parameter.")

args = parser.parse_args()

esmfold_contrib_energy(args.n_popul,args.sequence_to_mute,args.n_mut,args.initial,args.temp)

