import pandas as pd
#from tqdm import tqdm
from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation
pyrosetta.init()
import pickle
from Functions_Pyrosetta import Energy_contribution

def normalize_function(data,column):
    data_norm = pd.DataFrame()
    data_df = data.copy()
    data_norm[f'{column}'] = (data_df[f'{column}']-data_df[f'{column}'].min())
    data_norm[f'{column}'] = (data_norm[f'{column}']/(data_norm[f'{column}'].max()-data_norm[f'{column}'].min()))

    data[f'{column}_normalized'] = data_norm[f'{column}']

    return data

def Get_residues_from_chain(pose, chain):
    """
    Get the sequence and residue numbers for a specific chain in a given pose.

    Parameters:
    - pose: PyRosetta Pose object
    - chain: Chain identifier (e.g., 'A')

    Returns:
    A tuple containing the chain sequence and a list of residue numbers.
    """
    residue_numbers = [residue for residue in range(1, pose.size() + 1) if pose.pdb_info().chain(residue) == chain]
    chain_sequence = ''.join([pose.residue(residue).name1() for residue in residue_numbers])
    
    return chain_sequence, residue_numbers

def normalize_function_dG(data,column):

    data_norm = pd.DataFrame()
    data_df = data.copy()
    data_df[f'{column}'] = data_df[f'{column}']*-1
    data_norm[f'{column}'] = (data_df[f'{column}']-data_df[f'{column}'].min())
    data_norm[f'{column}'] = (data_norm[f'{column}']/(data_norm[f'{column}'].max()-data_norm[f'{column}'].min()))

    data[f'{column}_normalized_invert'] = data_norm[f'{column}']

    return data

def preparar_para_mutar(pdb_name,index_rosetta):

    ## variáveis
    energy_contribution_dF = pd.DataFrame()
    sequence_to_mute = ''

    structure_pose = f"PDBs/{pdb_name}.pdb"
    pose_sel_structure_ = pose_from_pdb(structure_pose)
    sequence_to_mute = pose_sel_structure_.sequence()
    index_rosetta_ = list(index_rosetta.iloc[:,0])

    energy_contribution_data = Energy_contribution(pose_sel_structure_, "False")
    energy_contribution_dF['cont_energy'] = energy_contribution_data.T
    contribuicao_teste = energy_contribution_dF.loc[index_rosetta_,:]
    contrib_energy_df = normalize_function(contribuicao_teste,'cont_energy')
    list_para_mutar = contrib_energy_df.reset_index()
    
    list_para_mutar['index'] = list_para_mutar['index'] - 1
    
    list_para_mutar.to_csv('dados_gerados/posi_to_mute.csv')

    return sequence_to_mute
