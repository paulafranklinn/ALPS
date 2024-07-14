#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:01:10 2023

@author: joao.sartori
"""
import argparse
import pandas as pd
import os
import random
import math
#imports from pyrosetta
from mimetypes import init
from pyrosetta import *
from pyrosetta.teaching import *
#from IPython.display import Image
#Core Includes
from rosetta.core.kinematics import MoveMap
from rosetta.core.kinematics import FoldTree
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation
from rosetta.core.simple_metrics import metrics
from rosetta.core.select import residue_selector as selections
from rosetta.core import select
from rosetta.core.select.movemap import *
from rosetta.protocols import minimization_packing as pack_min
from rosetta.protocols import relax as rel
from rosetta.protocols.antibody.residue_selector import CDRResidueSelector
from rosetta.protocols.antibody import *
from rosetta.protocols.loops import *
from rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.docking import setup_foldtree
from pyrosetta.rosetta.protocols import *
from rosetta.core.scoring.methods import EnergyMethodOptions

def read_pose(pdb):
    """
    Read a protein structure from a PDB file using PyRosetta.

    Parameters:
    - pdb (str): Path to the PDB file containing the protein structure.

    Returns:
    - pose (pyrosetta.rosetta.core.pose.Pose): PyRosetta pose object representing the protein structure.
    - scorefxn (pyrosetta.rosetta.core.scoring.ScoreFunction): PyRosetta score function used for energy calculations.

    Example:
    ```
    pdb_file = "example.pdb"
    protein_pose, energy_score_function = read_pose(pdb_file)
    ```

    Note:
    - The function initializes the PyRosetta framework using `pyrosetta.init()`.
    - It reads the protein structure from the specified PDB file using `pose_from_pdb`.
    - It creates a score function (`ref2015_cart.wts` in this case) using `pyrosetta.create_score_function`.
    - The energy of the structure is calculated using `scorefxn(pose)` to ensure proper initialization of the pose.

    Reference:
    - PyRosetta Documentation: https://graylab.jhu.edu/PyRosetta.documentation/

    """
    # Initialize PyRosetta
    pyrosetta.init()

    # Read the protein structure from the PDB file
    pose = pose_from_pdb(pdb)

    # Create a score function for energy calculations
    scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")

    # Calculate the energy of the structure to ensure proper initialization
    scorefxn(pose)

    return pose, scorefxn

def PDB_pose_dictionairy(pose_to):
    """
    Create a Pandas DataFrame representing the mapping of residue positions
    between PDB numbering and Pose numbering for a given PyRosetta pose.

    Parameters:
    - pose_to (PyRosetta pose): The PyRosetta pose object for which to create
                                the PDB to Pose numbering dictionary.

    Returns:
    - pandas.DataFrame: A DataFrame containing three columns - 'Chain',
                        'IndexPDB', and 'IndexPose' representing the mapping
                        of residue positions between PDB and Pose numbering.

    Example:
    >>> pose = ...  # Initialize your PyRosetta pose object
    >>> pdb_pose_dict = PDB_pose_dictionairy(pose)
    >>> print(pdb_pose_dict)
         Chain  IndexPDB  IndexPose
    0        A         1          1
    1        A         2          2
    ...      ...       ...        ...
    n        B       100         99
    """
    # List to store data during iteration
    vectorIndexChain = []
    vectorIndexPDB = []
    vectorIndexPose = []

    # Creating dictionary for residue position - PDB and Pose numbering
    for i in range(pose_to.total_residue()):
        vectorIndexChain.append(pose_to.pdb_info().chain(i + 1))
        vectorIndexPDB.append(pose_to.pdb_info().number(i + 1))
        vectorIndexPose.append(pose_to.pdb_info().pdb2pose(vectorIndexChain[i], vectorIndexPDB[i]))

    # Inserting values into a pandas dataframe
    df_pdbtopose_dictionary = {"Chain": vectorIndexChain,
                                "IndexPDB": vectorIndexPDB,
                                "IndexPose": vectorIndexPose}
    df_dictionary = pd.DataFrame(df_pdbtopose_dictionary)

    return df_dictionary

def residues_list(df, chain):
    """
    Extract a list of Pose indices corresponding to a specific chain from a
    DataFrame containing PDB to Pose numbering mapping.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing PDB to Pose numbering mapping,
                             typically generated by the PDB_pose_dictionairy function.
    - chain (str): The chain identifier for which to extract the Pose indices.

    Returns:
    - list: A list of Pose indices corresponding to the specified chain.

    Example:
    >>> pdb_pose_dict = ...  # DataFrame generated using PDB_pose_dictionairy
    >>> chain_A_residues = residues_list(pdb_pose_dict, 'A')
    >>> print(chain_A_residues)
    [1, 2, ..., n]
    """
    # Extracting a list of Pose indices for the specified chain
    lists = list(df[df['Chain'] == chain]['IndexPose'])
    return lists

def pack_relax(pose, scorefxn):
    """
    Perform a fast relaxation protocol on the given pose using PyRosetta's FastRelax.

    Parameters:
    - pose: PyRosetta Pose object
    - scorefxn: Score function to evaluate the energy of the structure

    Returns:
    None
    """
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    # Set up a MoveMapFactory
    mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
    mmf.all_bb(setting=True)
    mmf.all_bondangles(setting=True)
    mmf.all_bondlengths(setting=True)
    mmf.all_chi(setting=True)
    mmf.all_jumps(setting=True)
    mmf.set_cartesian(setting=True)

    ## Print informations about structure before apply fast relax
    # display_pose = pyrosetta.rosetta.protocols.fold_from_loops.movers.DisplayPoseLabelsMover()
    # display_pose.tasks(tf)
    # display_pose.movemap_factory(mmf)
    # display_pose.apply(pose)

    fr = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn_in=scorefxn, standard_repeats=1)
    fr.cartesian(True)
    fr.set_task_factory(tf)
    fr.set_movemap_factory(mmf)
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.apply(pose)
    return 

def mutate_repack(starting_pose, posi, amino, scorefxn):
    """
    Mutate a specific position in a pose, select neighboring positions, and repack the residues.
    
    Parameters:
    - starting_pose: PyRosetta Pose object
    - posi: Position to mutate
    - amino: Amino acid to mutate to
    - scorefxn: Score function to evaluate the energy of the structure
    
    Returns:
    A new Pose object after mutation and repacking.
    """
    pose = starting_pose.clone()
    
     #Select position to mutate
    mut_posi = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    mut_posi.set_index(posi)
    
    #Select neighbor positions
    nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(mut_posi)
    nbr_selector.set_include_focus_in_subset(True)
    
    not_design = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(mut_posi)

    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()

    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    # Disable Packing
    prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
    prevent_subset_repacking = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking_rlt, nbr_selector, True )
    tf.push_back(prevent_subset_repacking)

    # Disable design
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(),not_design))

    # Enable design
    aa_to_design = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    aa_to_design.aas_to_keep(amino)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(aa_to_design, mut_posi))

    # Create Packer
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
    packer.task_factory(tf) 
    packer.apply(pose)
    
    return pose

def unbind(pose, partners, scorefxn):
    """
    Simulate unbinding by applying a rigid body translation to a dummy pose and performing FastRelax.

    Parameters:
    - pose: PyRosetta Pose object representing the complex
    - partners: Vector1 specifying the interacting partners
    - scorefxn: Score function to evaluate the energy of the structure

    Returns:
    A tuple containing two Pose objects, one for the unbound state and one for the bound state.
    """
    #### Generates dummy pose to maintain original pose
    pose_dummy = pose.clone()
    pose_binded = pose.clone()
    STEP_SIZE = 100
    JUMP = 1
    docking.setup_foldtree(pose_dummy, partners, Vector1([-1,-1,-1]))
    trans_mover = rigid.RigidBodyTransMover(pose_dummy,JUMP)
    trans_mover.step_size(STEP_SIZE)
    trans_mover.apply(pose_dummy)
    pack_relax(pose_dummy, scorefxn)
    #### Return a tuple containing:
    #### Pose binded = [0] | Pose separated = [1]
    return pose_binded , pose_dummy


def dG_v2_0(pose_Sep, pose_bind, scorefxn):
    """
    Calculate the binding energy difference between the unbound and bound states.

    Parameters:
    - pose_Sep: Pose object representing the unbound state
    - pose_bind: Pose object representing the bound state
    - scorefxn: Score function to evaluate the energy of the structure

    Returns:
    The binding energy difference (dG).
    """
    bound_score = scorefxn(pose_bind)
    unbound_score = scorefxn(pose_Sep)
    dG = bound_score - unbound_score
    return dG

def Dg_bind(pose, partners, scorefxn):
    """
    Calculate the binding energy difference for a given complex.

    Parameters:
    - pose: PyRosetta Pose object representing the complex
    - partners: Vector1 specifying the interacting partners
    - scorefxn: Score function to evaluate the energy of the structure

    Returns:
    The binding energy difference (dG) for the complex.
    """
    pose_dummy = pose.clone()
    unbinded_dummy = unbind(pose_dummy, partners, scorefxn)
    #unbinded_dummy[1].dump_pdb("./Dumps/unbinded_teste.pdb")
    #unbinded_dummy[0].dump_pdb("./Dumps/binded_teste.pdb")
    return (dG_v2_0(unbinded_dummy[1], unbinded_dummy[0], scorefxn))


def Get_residues_from_pose(pose):
    """
    Get the sequence and residue numbers for a specific chain in a given pose.

    Parameters:
    - pose: PyRosetta Pose object


    Returns:
    A tuple containing the sequence and a list of residue numbers.
    """
    
    residue_numbers = [residue for residue in range(1, pose.size() + 1)]
    sequence = ''.join([pose.residue(residue).name1() for residue in residue_numbers])

    return sequence, residue_numbers

def Compare_sequences(before_seq, after_seq, indexes):
    """
    Compare two sequences and identify mutations. Print new mutations.

    Parameters:
    - before_seq: Original sequence
    - after_seq: Mutated sequence
    - indexes: Residue indexes

    Returns:
    A dictionary containing the mutated residues.
    """
    wt = before_seq
    mut = after_seq
    
    mutation = dict()
    for index, (res1, res2) in enumerate(zip(wt, mut)):
        if res1 != res2:
            mutation[indexes[index]] = res2
            print(f"New mutation \U0001f600 : {res1}{indexes[index]}{res2}")
    return mutation

def model_sequence(pose, mutations, scorefxn, relax = True):
    """
    Model a sequence on a given pose by applying mutations and repacking.

    Parameters:
    - pose: PyRosetta Pose object
    - mutations: Dictionary of mutations
    - scorefxn: Score function to evaluate the energy of the structure
    - relax: bool, optional, default=True : Whether to apply a relax protocol to the pose after mutations and repacking.
    Returns:
    A new Pose object after modeling the sequence.
    """
    new_pose = pose.clone()
    to_mutate = mutations
    
    for index in to_mutate:
        new_pose = mutate_repack(starting_pose = new_pose, posi = index, amino = to_mutate[index], scorefxn = scorefxn)
    if relax:
        pack_relax(pose = new_pose, scorefxn = scorefxn)
    return new_pose


def Model_structure(pdb, sequence, output_name, relax = True):
    """
    Perform protein sequence modeling using PyRosetta based on input data.

    Parameters:
    - pdb (str): Path to the PDB file containing the initial protein structure.
    - sequence (str): Variable containing the sequence.
    - output_name (str): Output name for the modeled structure
    - relax: bool, optional, default=True : Whether to apply a relax protocol to the pose after mutations and repacking.
    Returns:
    - new_pose (pyrosetta.rosetta.core.pose.Pose): PyRosetta pose object representing the protein structure.
    - pdb file: The structure in pdb format.
    Example:
    ```
    pdb_file = "initial_structure.pdb"
    sequences_to_model = "AAAAAA"

    ```

    Note:
    - The function initializes the PyRosetta framework by calling the `read_pose` function.
    - It reads the initial protein structure and creates a clone for subsequent modeling.
    - It compares the current structure with the target sequence.
    - Mutations needed to transform the current structure into the target sequence are determined.
    - The sequence is modeled using the `model_sequence` function, and the resulting data is stored in pose.

    Reference:
    - PyRosetta Documentation: https://graylab.jhu.edu/PyRosetta.documentation/

    """
    pose, scorefxn = read_pose(pdb)
    pose_init = pose.clone()
    #### Reads input file with sequences to be modeled
    
    residues_from_chain, index = Get_residues_from_pose(pose = pose)
    mutations = Compare_sequences(before_seq = residues_from_chain, after_seq = sequence, indexes = index)
    new_pose = model_sequence(pose_init, mutations, scorefxn, relax)
    
    new_pose.dump_pdb(f"{output_name}.pdb")
    return new_pose

def Energy_contribution(pose, by_term):
    """
    Calculate and analyze the energy contributions of different terms for each residue in a given protein pose.

    Parameters:
    - pose: PyRosetta Pose object representing the protein structure.
    - by_term: Boolean flag indicating whether to analyze energy contributions by term (True) or by residue (False).

    Returns:
    - DataFrame: If by_term is True, returns a DataFrame containing energy contributions by term for each residue.
                 If by_term is False, returns a DataFrame containing energy contributions by residue.
    """

    # List of energy terms from ref2015_cart
    listadosdicts = [fa_atr, fa_rep, fa_sol, fa_intra_rep, fa_intra_sol_xover4,
                lk_ball_wtd, fa_elec, hbond_sr_bb, hbond_lr_bb, hbond_bb_sc, hbond_sc, dslf_fa13,
                omega, fa_dun, p_aa_pp, yhh_planarity, ref, rama_prepro, cart_bonded]
    
    # Create a score function using ref2015_cart weights
    scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
    weights = scorefxn.weights()
    
    # Set up energy method options to decompose hbond terms
    emopts = EnergyMethodOptions(scorefxn.energy_method_options())
    emopts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    scorefxn.set_energy_method_options(emopts)
    
    # Calculate energy scores for the given pose
    scorefxn.score(pose)
    
    # Check if the user wants to analyze energy contributions by term
    if by_term == True:
        # Initialize a dictionary for storing data
        dasd = {'Protein': [], 'Sequence': []}
        
        # Get all residues' pose index from pose
        Residues = [residue.seqpos() for residue in pose]
        dasd['Protein'].append("WT")
        
        # Populate the dictionary with energy contributions for each residue and term
        for posi in Residues:
            for i in range(len(listadosdicts)):
                term_key = '{}-%s'.format(posi) % listadosdicts[i]
                dasd[term_key] = []
                dasd[term_key].append(pose.energies().residue_total_energies(posi)[listadosdicts[i]])
        dasd['Sequence'].append(pose.sequence())
        
        # Create a DataFrame from the dictionary
        df2 = pd.DataFrame(dasd)
        
        # Create a DataFrame with energy terms and their respective weights
        weights_by_term = pd.DataFrame(index=range(1, len(listadosdicts)+1), columns=range(0, 2))
        weights_by_term.iloc[:, 0] = listadosdicts
        list_weights = [1, 0.55, 1, 0.005, 1, 1, 1, 1, 1, 1, 1, 1.25, 0.4, 0.7, 0.6, 0.625, 1, 0.45, 0.5]
        weights_by_term.iloc[:, 1] = list_weights
        
        # Apply weights to each term in the energy contribution DataFrame
        for i in range(len(weights_by_term)):
            list_to_change = df2.filter(like=str(weights_by_term.iloc[i, 0])).columns
            df2[list_to_change] = df2[list_to_change] * weights_by_term.iloc[i, 1]
        
        return df2
    else:
        # If not analyzing by term, create a DataFrame with energy contributions by residue
        seq_size = len([x for x in pose.sequence()])
        Residues = [residue.seqpos() for residue in pose]
        df_byresidue = pd.DataFrame(index=range(1, 2), columns=range(1, seq_size+1))
        
        for i in range(1, len(df_byresidue.columns)+1):
            df_byresidue.iloc[0, i-1] = pose.energies().residue_total_energy(Residues[i-1])
        
        return df_byresidue
