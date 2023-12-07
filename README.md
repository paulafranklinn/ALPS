# peptide machine

# Usage

## function_peptide.R 

Função que recorta peptídeos de interesse, usando como métrica a distância de residuos de interesse.

## Gerador_de_seq

Gera uma população de sequencias únicas.

## modeling_functions_pyrosetta.py

Python set of functions used to model sequences and output the dG values into a Pandas dataframe and .csv file. 
Reads a dataframe containing the sequences and a PDB to use the structure as a template. Example input and output file can be found in the "testing_data" folder.

## Feature_extraction

Converte peptídeos em descritores

## Modelo_MLR_peptide

Tem funções de pre-processamento, seleção do numero de k para o select k best e um modelo de regressão linear multipla (lasso).

