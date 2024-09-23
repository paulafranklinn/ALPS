#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:07:42 2024

@author: joao
"""

import os
import pandas as pd
import time
import subprocess
from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation
from tqdm import tqdm
import os
import multiprocessing
import pandas as pd
from tqdm import tqdm
import time
from Modeling_functions_pyrosetta import Execute

def Run_all_batches(pose, list_seq,batch_indexes,cycle):
    pose_init = pose.clone()
    scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
    
    processes = []

    for x in tqdm(range(len(batch_indexes)), desc="Processing list of sequences"):
        p = multiprocessing.Process(target=Execute,args=[pose, scorefxn,list_seq[x],batch_indexes[x],cycle])
        p.start()
        processes.append(p)
            
    for p in processes:
        p.join()
        
def remove_files(file_list):
    for file_name in file_list:
        try:
            os.remove(file_name)
            print(f"File '{file_name}' removed successfully.")
        except OSError as e:
            print(f"Error removing file '{file_name}': {e}")


def check_files_in_directory(directory, file_list):
    """
    Check if all files in the given list exist in the specified directory.

    Args:
    directory (str): The directory to check for the files.
    file_list (list): A list of filenames to check for.

    Returns:
    bool: True if all files exist in the directory, False otherwise.
    """
    while True:
        # Check if the directory exists
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            time.sleep(10)  # Wait for 10 seconds before retrying
            continue

        # Check if all files exist in the directory
        missing_files = []
        for filename in file_list:
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                missing_files.append(filename)
        
        if not missing_files:
            # All files exist in the directory
            to_save = retrieve_data(directory, file_list)
            remove_files(file_list)
            return to_save

        else:
            print(f"Not all files exist in directory '{directory}'. Missing files: {missing_files}")
            time.sleep(10)  # Wait for 10 seconds before retrying

def retrieve_data(directory, file_list):
    sequence = []
    dG = []
    for file in file_list:
        print(f"{file}")
        df_temp = pd.read_csv(f"{file}")
        sequence.append(df_temp.iloc[0,1])
        dG.append(df_temp.iloc[0,2])
    return sequence, dG

def run_python_script(script_path, param1, param2):
    try:
        subprocess.run(['python', script_path, param1, param2], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")


def batchs_to_run(pose,sequences,batch_size,cycle):
    list_seq_final = []
    list_dg_final = []

    remaining_sequences = sequences[:]# Make a copy of the sequences
    batch_index = 0  # Initialize batch index
    while remaining_sequences:
        lista_seq = []
        lista_dg = []
        # Take a batch of sequences from the remaining ones
        batch = remaining_sequences[:batch_size]
        print(batch)
        #list_files_temp = [f"temp_{i}.csv" for i in batch]
        list_files_temp = [f"temp_{batch_index * batch_size + i}.csv" for i in range(len(batch))]  # Adjusted line
        batch_indexes = [batch_index * batch_size + i for i in range(len(batch))] 
        print(batch_indexes)
        remaining_sequences = remaining_sequences[batch_size:]  # Update remaining sequences
        # Process the current batch (here, just print the sequences)
        print("Processing batch:")
        Run_all_batches(pose, batch, batch_indexes,cycle)
        ### Check it all files in list temp are in dir, if true, get sequences and dG from all 
        seqs, dgs = check_files_in_directory(".", list_files_temp)
        lista_seq.append(seqs)
        lista_dg.append(dgs)
        

        # Append the current batch size to the list
        list_seq_final += lista_seq
        list_dg_final += lista_dg
        time.sleep(5)
        batch_index += 1  # Increment batch index
    seqs_finais_ficou_de_verdade = [i for lista in list_seq_final for i in lista]
    dgs_finais_ficou_de_verdade = [i for lista in list_dg_final for i in lista]
    return seqs_finais_ficou_de_verdade, dgs_finais_ficou_de_verdade


# pyrosetta.init()

# estrutura_pdb = "3mre_recortada_relax.pdb"
# pose = pose_from_pdb(estrutura_pdb)
# scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")

# list_seq = ['ALCTLVAML','GACTLVAML','GLATLVAML','GLCALVAML','GLCTAVAML','GLCTLAAML','GCCTLVAML','GLCTLVAAL','GLCTTVAML','GLCTLVAFL','GFCTLVAML','FLCTLVAML','GLCTLVAMF','GLCTLFAML','GLCVLVAML','GLVTLVAML','GLCTLVVML']


# seqs, dgs = batchs_to_run(pose, list_seq, 6)

