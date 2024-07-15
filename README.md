# Peptide Machine

Welcome to the Peptide Machine project! This repository contains all the necessary scripts and instructions to run the Peptide Machine. Please follow the instructions below to set up your environment and run the script.

## Table of Contents

- [Installation](#installation)
  - [Using .yml File](#using-yml-file)
  - [Installing Rosetta](#installing-rosetta)
  - [Installing PyRosetta](#installing-pyrosetta)
  - [Pbee](#Pbee)
- [Running the Script](#running-the-script)
- [Resources](#resources)

## Installation

### Using .yml File

To set up your environment using the provided `.yml` file, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/paulafranklinn/ALPS.git
    cd peptide-machine
    ```

2. Create the environment:
    ```sh
    conda env create -f environment.yml
    ```

3. Activate the environment:
    ```sh
    conda activate ALPS-env
    ```

### Installing Rosetta

For instructions on how to install Rosetta, refer to this [YouTube video](https://www.youtube.com/watch?v=UEaFmUMEL9c&t=16s).

### Installing PyRosetta

Download and install PyRosetta from [this link](https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python310.linux/).

### Pbee

To measure the binding free energy, we use the Protein Binding Energy Estimator (Pbee). Therefore, it is necessary to add it to the same folder as this repository. It can be obtained from this [link](https://github.com/chavesejf/pbee).

## Running the Script

Once you have your environment set up, you can edit and run the Peptide Machine script using the following command:

```sh
python script_mestre.py --estrutura_pdb /BACKUP/dados/Projetos/Mestrado/scripts_atualizados/Mestrado/estruturas_iniciais/3mre_recortada_relax.pdb \
                     --n_loop 2 \
                     --seq_numb 8 \
                     --replica 0 \
                     --list_residue [1 2 3 4 5 6 7 8 9] \
                     --list_chain P \
                     --chain A_P \
                     --cpu 4 \
                     --model RF
