#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:44:34 2023

@author: joao.sartori
"""


#### Importing rosetta functions
from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation

#### Importing secondary libs
import os
import numpy as np
from numpy.random import uniform
from random import sample
import pandas as pd

pyrosetta.init()


#### Function to mutate a specific residue
def mutate_repack(starting_pose, posi, amino):
    
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

#### Iterates through input sequence, model and return dG value
def model_sequence(pose, sequence):
    sequence_full = [resid for resid in sequence]
    for i in range(len(sequence_full)):
        pose = mutate_repack(pose, posi=i+1, amino=sequence_full[i])
    return scorefxn(pose)

#### Use template structure and sequences to model and extract its dG value into a pandas dataframe
#### and outputs a .csv file with the results
def Execute(pose, input_csv):
    pose_init = pose.clone()
    #### Reads input file with sequences to be modeled
    seqs = input_csv
    data = pd.DataFrame(index = range(1,len(seqs.index)+1), columns = range(1,3))
    for i in range(0, len(data.index)):
        sequence = seqs.iloc[i,0]
        pose = pose_init
        dG = model_sequence(pose, sequence)
        data.iloc[i,0] = ''.join(sequence)
        data.iloc[i,1] = dG
    data.to_csv("output.csv") 
    return data
    
     
#### Read input PDB structure used for construct the models
pose = pose_from_pdb("2lzt.pdb")
#### Define score ref2015_cart and apply it to the pose
scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
scorefxn(pose)
#### Reads input file with sequences to be modeled
inputcsv = pd.read_csv("input.csv")
 
#### Execute main function to model sequences and extract dG values
dG_modeled_sequences = Execute(pose, inputcsv)





