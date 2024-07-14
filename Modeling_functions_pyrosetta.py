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
import time
import numpy as np
from numpy.random import uniform
#imports from pyrosetta
from mimetypes import init
from pyrosetta import *
from pyrosetta.teaching import *
import subprocess
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


#### FastRelax Protocol - bb_unlocked
####
# def pack_relax(pose, scorefxn):
#     """
#     Perform a fast relaxation protocol on the given pose using PyRosetta's FastRelax.

#     Parameters:
#     - pose: PyRosetta Pose object
#     - scorefxn: Score function to evaluate the energy of the structure

#     Returns:
#     None
#     """
#     tf = TaskFactory()
#     tf.push_back(operation.InitializeFromCommandline())
#     tf.push_back(operation.RestrictToRepacking())
#     # Set up a MoveMapFactory
#     mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
#     mmf.all_bb(setting=True)
#     mmf.all_bondangles(setting=True)
#     mmf.all_bondlengths(setting=True)
#     mmf.all_chi(setting=True)
#     mmf.all_jumps(setting=True)
#     mmf.set_cartesian(setting=True)

#     ## Print informations about structure before apply fast relax
#     # display_pose = pyrosetta.rosetta.protocols.fold_from_loops.movers.DisplayPoseLabelsMover()
#     # display_pose.tasks(tf)
#     # display_pose.movemap_factory(mmf)
#     # display_pose.apply(pose)

#     fr = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn_in=scorefxn, standard_repeats=1)
#     fr.cartesian(True)
#     fr.set_task_factory(tf)
#     fr.set_movemap_factory(mmf)
#     fr.min_type("lbfgs_armijo_nonmonotone")
#     fr.apply(pose)
#     return 

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
    #pack_relax(pose_dummy, scorefxn,1)
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
    #bound_score = mc(pose_bind, scorefxn, 1000, 10)
    #unbound_score = mc(pose_Sep, scorefxn, 1000, 10)

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
            #print(f"New mutation \U0001f600 : {res1}{indexes[index]}{res2}")
    return mutation


def model_sequence(pose, mutations, scorefxn,chain):
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
    #new_pose = pack_relax(new_pose,scorefxn,1)
    #dg = Dg_bind(new_pose,chain,scorefxn)
    return new_pose

def Get_residues_from_pose(pose):
    """
    Get the sequence and residue numbers for a specific chain in a given pose.

    Parameters:
    - pose: PyRosetta Pose object
    - chain: Chain identifier (e.g., 'A')

    Returns:
    A tuple containing the chain sequence and a list of residue numbers.
    """
    
    residue_numbers = [residue for residue in range(1, pose.size() + 1)]
    sequence = ''.join([pose.residue(residue).name1() for residue in residue_numbers])

    # residue_numbers = [residue for residue in range(1, pose.size() + 1) if pose.pdb_info().chain(residue) == chain]
    # chain_sequence = ''.join([pose.residue(residue).name1() for residue in residue_numbers])
    
    return sequence, residue_numbers

def Execute(pose,scorefxn,sequence,i,chain,cycle):
    
    print('Execute esta rodando')
    
    """
    Perform protein sequence modeling using PyRosetta based on input data.

    Parameters:
    - pdb (str): Path to the PDB file containing the initial protein structure.
    - dataframe (pd.DataFrame): DataFrame containing sequences to be modeled.
    - chain (str): Chain identifier for the protein structure.

    Returns:
    - data (pd.DataFrame): DataFrame containing modeled sequences and their corresponding dG values.
    
    Example:
    ```
    pdb_file = "initial_structure.pdb"
    sequences_to_model = pd.DataFrame({"Sequence": ["AAABBB", "CCCDDD", "EEFFGG"]})
    chain_id = "A"
    modeling_results = Execute(pdb_file, sequences_to_model, chain_id)
    ```

    Note:
    - The function initializes the PyRosetta framework by calling the `read_pose` function.
    - It reads the initial protein structure and creates a clone for subsequent modeling.
    - For each sequence in the input DataFrame, it compares the current structure with the target sequence.
    - Mutations needed to transform the current structure into the target sequence are determined.
    - The sequence is modeled using the `model_sequence` function, and the resulting data is stored in a DataFrame.
    - The execution time is recorded, and the results are saved in CSV and text files.

    Reference:
    - PyRosetta Documentation: https://graylab.jhu.edu/PyRosetta.documentation/

    """
    
    pose_init = pose.clone()
    #### Reads input file with sequences to be modeled
    
    residues_from_chain, index = Get_residues_from_pose(pose = pose)
    mutations = Compare_sequences(before_seq = residues_from_chain, after_seq = sequence, indexes = index)
    new_pose = model_sequence(pose_init, mutations,scorefxn,chain)
    new_pose.dump_pdb(f"PDBs/{cycle}_{i}.pdb")

    command = f"python3 pbee.py --ipdb ./PDBs/{cycle}_{i}.pdb --partner1 A --partner2 P --odir . --force_mode"
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    temp_df = pd.read_csv(f"outputs_pbee/{cycle}_{i}/dG_pred.csv")
    score = temp_df["dG_pred"][0]
    #### linha para ler o output csv, salvar o dG na variavel score e seguir com a vida
    #score = Dg_bind(new_pose, "A_D", scorefxn)
    data = pd.DataFrame({'Sequence': [new_pose.sequence()],'dG': [score],'seq_numb': [i]})
    data.to_csv(f'temp_{i}.csv')        
               
    return data

def decision(before_pose, after_pose, scorefxn):
    ### BEGIN SOLUTION
    E = scorefxn(after_pose) - scorefxn(before_pose)
    if E < 0:
        return after_pose
    elif random.uniform(0, 1) >= math.exp(-E/1):
        return before_pose
    else:
        return after_pose
    ### END SOLUTION
    
def mc(pose_ref, scorefxn, en_size, en_runs):

    logging.basicConfig(filename='mc.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting Monte Carlo simulation")
    en_size=en_size ## ensemble_size
    kT=1.0
    n_moves=2 ### number of shear mover movements
    anglemax_ss=1 ### anglemax of shearmove in Alpha helix and beta-sheets
    anglemax=1 ### anglemax of shearmove in coil regions

    structure = pose_ref.clone()
    score = []
    score_final = []
    movemap = MoveMap()

    to_move = list(range(1, len(pose_ref.sequence())))

    move_map_list=[]
    for posi in to_move:
        mut_posi = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
        mut_posi.set_index(posi)

        # Select Neighbor Position
        nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
        nbr_selector.set_focus_selector(mut_posi)
        nbr_selector.set_include_focus_in_subset(True)
        bla=pyrosetta.rosetta.core.select.get_residues_from_subset(nbr_selector.apply(pose_ref))
        for xres in bla:
            move_map_list.append(xres)


    movemap = MoveMap()
    for x in move_map_list:
        movemap.set_bb(x,True)


    minmover = MinMover()
    minmover.movemap(movemap)
    minmover.score_function(scorefxn)


    ###################
    ## Version 4 changes
    ##################
    to_pack = standard_packer_task(structure)
    to_pack.restrict_to_repacking()    # prevents design, packing only
    to_pack.or_include_current(True)    # considers the original sidechains


    to_pack.temporarily_fix_everything()

    for xres in bla:
        to_pack.temporarily_set_pack_residue(xres, True)

    packmover = PackRotamersMover(scorefxn, to_pack)


    shearmover = ShearMover(movemap, kT, n_moves) ## move phi e psi do mesmo residuo


    shearmover.angle_max('H', anglemax_ss) ##helices
    shearmover.angle_max('E', anglemax_ss) ## strands
    shearmover.angle_max('L', anglemax) ## loops

    combined_mover = SequenceMover()
    combined_mover.add_mover(shearmover)
    combined_mover.add_mover(packmover)

    before_pose = pose_ref.clone()
    for ensembles in range(en_runs):
        logging.info(f"Starting Monte Carlo simulation: run {ensembles}")
        for en in range(en_size):

            after_pose = before_pose.clone()

            #### APPLY
            combined_mover.apply(after_pose)
            #structure.dump_pdb("./ensembles/rbd_shear_"+str(en)+"_"+name_debug+".pdb")

            # The task factory accepts all the task operations

            #after_pose.dump_pdb("Dumps_ensemble/controle_mc_"+str(en)+".pdb")
            score.append(scorefxn(after_pose))

            before_pose = decision(before_pose, after_pose, scorefxn)

            #structure.dump_pdb("./ensembles/rbd_decision_"+str(en)+"_"+name_debug+".pdb")

            ###dump for debug
           #before_pose.dump_pdb("Dumps_ensemble/teste_"+str(en)+".pdb")

            score_final.append(scorefxn(before_pose))

    return np.mean(score_final)
    

def pack_relax(starting_pose, scorefxn, times_to_relax):
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

    fr = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn_in=scorefxn, standard_repeats=times_to_relax)
    fr.cartesian(True)
    fr.set_task_factory(tf)
    fr.set_movemap_factory(mmf)
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.apply(starting_pose)

    #starting_pose.dump_pdb("3mre_recortada_relax.pdb")
    #starting_pose.dump_pdb("CD19_scFv_relax.pdb")

    return starting_pose
