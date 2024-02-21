#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:46:57 2024

@author: papaula
"""

import pandas as pd

#### Importing rosetta functions
from pyrosetta import *
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation
pyrosetta.init()

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

def resi_index_PDB_to_rosetta(pose,list_chain,list_residue):
    
    df_index = PDB_pose_dictionairy(pose)
    y = pd.DataFrame()
    for k in range(len(list_chain)):
        for o in range(len(list_residue[k])):
            mask = (df_index["Chain"] == list_chain[k]) & (df_index["IndexPDB"] == list_residue[k][o])
            subset_df = df_index.loc[mask, ["IndexPose"]]
            y = pd.concat([y, subset_df])
            
    index_rosetta = y.reset_index(drop=True)
    return index_rosetta