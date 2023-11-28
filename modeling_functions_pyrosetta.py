#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:41:38 2023

@author: joao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:30:38 2023

@author: joao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:33:21 2023

@author: joao
"""

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
#### Importing secondary libs
import os
import numpy as np
from numpy.random import uniform


#### FastRelax Protocol - bb_unlocked
####
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


def model_sequence(pose, mutations, scorefxn):
    """
    Model a sequence on a given pose by applying mutations and repacking.

    Parameters:
    - pose: PyRosetta Pose object
    - mutations: Dictionary of mutations
    - scorefxn: Score function to evaluate the energy of the structure

    Returns:
    A new Pose object after modeling the sequence.
    """
    new_pose = pose.clone()
    to_mutate = mutations
    
    for index in to_mutate:
        new_pose = mutate_repack(starting_pose = new_pose, posi = index, amino = to_mutate[index], scorefxn = scorefxn)
    #new_pose = pack_relax(pose = new_pose, scorefxn = scorefxn)
    return new_pose, scorefxn(new_pose)

def Execute(pose, dataframe, chain, scorefxn):
    """
    Execute the entire modeling pipeline for a set of sequences in a DataFrame, saving the results to a CSV file.

    Parameters:
    - pose: PyRosetta Pose object
    - dataframe: Input DataFrame with sequences
    - chain: Chain identifier (e.g., 'A')
    - scorefxn: Score function to evaluate the energy of the structure

    Returns:
    A DataFrame containing the modeled sequences and their binding energies.
    """
    pose_init = pose.clone()
    #### Reads input file with sequences to be modeled
    data = pd.DataFrame(index = range(1,len(dataframe.index)+1), columns = range(1,3))
    for i in range(0, len(dataframe.index)):
        sequence = dataframe.iloc[i,0]
        residues_from_chain, index = Get_residues_from_chain(pose = pose, chain = chain)
        mutations = Compare_sequences(before_seq = residues_from_chain, after_seq = sequence, indexes = index)
        # sequence_to_compare, indexs = Get_residues_from_chain(pose = pose, chain = chain)
        # sequence_to_model = Get_residues_from_indexs(sequence = sequence, indexs = indexs)
        new_pose, dG = model_sequence(pose_init, mutations, scorefxn)
        data.iloc[i,0] = ''.join(new_pose.sequence())
        data.iloc[i,1] = dG
    data.to_csv("output.csv") 
    return data
    


