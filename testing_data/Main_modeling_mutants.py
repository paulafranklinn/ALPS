#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:43:04 2023

@author: joao
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
import modeling_functions_pyrosetta


pyrosetta.init()


#### Read input PDB structure used for construct the models
pose = pose_from_pdb("7t9k_alfa1_rbd_relaxed.pdb")
#### Define score ref2015_cart and apply it to the pose
scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
scorefxn(pose)


decoys_Seqs = pd.DataFrame(index = range(1,4), columns = range(1,2))

decoys_Seqs.iloc[0,0] = "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTAAAA"
decoys_Seqs.iloc[1,0] = "AAAAEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITE"
decoys_Seqs.iloc[2,0] = "QQQQQQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITE"


modeling_functions_pyrosetta.Execute(pose = pose, dataframe = decoys_Seqs, chain = "D", scorefxn = scorefxn)
