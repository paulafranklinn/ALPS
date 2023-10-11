library(bio3d)
library(data.table)

Recorte_func <- function(pdb_code, target_chain, resi, cutoff){

## variaveis:
ptn <- read.pdb(pdb_code)
ptn_atom <- ptn$atom
residue_list <- resi
m <- 0

## separação dos ligantes e alvo
target_ptn <- ptn_atom[which(ptn_atom$chain == target_chain & ptn_atom$elety == "CA" & ptn_atom$type == "ATOM" & ptn_atom$resno %in% residue_list),]
ligand_ptn <- ptn_atom[which(ptn_atom$chain != target_chain & ptn_atom$type == "ATOM"),]


## Calculo de átomos em um raio (cutoff)
for (i in 1:nrow(target_ptn)) {
  x1 <- target_ptn[i,9]
  y1 <- target_ptn[i,10]
  z1 <- target_ptn[i,11]
  
  for (j in 1:nrow(ligand_ptn)) {
    x2 <- ligand_ptn[j,9]
    y2 <- ligand_ptn[j,10]
    z2 <- ligand_ptn[j,11]
    
    d <- sqrt((x1 - x2)**2+(y1 - y2)**2+(z1 - z2)**2)
    if(d <= cutoff){
        m[j] <- ligand_ptn$eleno[j]
    }
  }
}

m<- na.omit(m)

lig_pot <- ptn_atom[which(ptn_atom$chain != target_chain & ptn_atom$eleno %in% m),]

index = unique(lig_pot$resno)

# Cria lista com resíduos que são ligantes potenciais dado o cutoff selecionado
vec <- index
# Find the differences between consecutive elements
diff_vec <- c(1, diff(vec))
# Use cumsum to create groups of contiguous values
groups <- cumsum(diff_vec != 1)
# Split the vector into a list of sequences
split_vec <- split(vec, groups)
# Remove empty sequences
split_vec <- split_vec[sapply(split_vec, length) > 0]
# Print the resulting list of sequences
split_vec

## Cria um arquivo pdb para potenciais ligantes peptídicos
for (k in 1:length(split_vec)){
  lig_potenciais <- lig_pot[which(lig_pot$resno %in% split_vec[[k]]),]
  chain_pdb <- unique(lig_potenciais$chain)

  index = unique(lig_potenciais$resno)
  lig_resi_pept <- trim.pdb(ptn, resno= index, chain = chain_pdb)

  seq_min <- min(split_vec[[k]])
  seq_max <- max(split_vec[[k]])

  write.pdb(lig_resi_pept, file = paste(getwd(),"/",seq_min,"_",seq_max,"_",chain_pdb,"_",pdb_code,".pdb",sep = ""))
}
}
