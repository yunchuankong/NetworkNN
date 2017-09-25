setwd("C:\\Users\\kyccw\\Dropbox\\Research_Yu\\hurricane")

## convert gene names for the network data 
## use R 3.4.0 or above, do not use MRO
# library(org.Hs.eg.db)
# dat <- as.matrix(read.table("gene_network.txt", header=T))
# 
# for (i in 1:dim(dat)[1]) {
# # for (i in 1:100) {
# 	for (j in 1:2) {
# 		temp <- mget(dat[i,j], org.Hs.egSYMBOL2EG, ifnotfound=NA)[[1]]
# 		if (length(temp) == 1) {
# 			dat[i,j] <- temp
# 		} else {
# 			dat[i,j] <- temp[1]
# 		}
# 		
# 	}
# 	if (i %% 100 == 0) {
# 		cat(i,"\n")
# 	}
# }
# 
# edgelist <- dat[!is.na(dat[,1]),]
# edgelist <- edgelist[!is.na(edgelist[,2]),]
# 
# write.table(edgelist,"edgelist.txt")

## 
load("BRCA.tumor.bin")
labels <- as.matrix(read.csv("labels.csv", header=T))

# match expression data and labels
labels[which(labels[,1]=="Negative"),1] <- 0
labels[which(labels[,1]=="Positive"),1] <- 1
# labels[which(labels[,1]=="LIVING"),1] <- 0
# labels[which(labels[,1]=="DECEASED"),1] <- 1
# labels[which(as.numeric(labels[,1])<24),1] <- 0
# labels[which(as.numeric(labels[,1])>=24),1] <- 1

# all ID length is 12 (checked)
# for (i in 1:nrow(labels)) {
#   if (length(unlist(strsplit(labels[i,2],""))) != 12) {
#     cat(i,"\n")
#   }
# }

## approach 1
# names_s <- colnames(array)
# for (i in 1:ncol(array)) {
#   names_s[i] <- fetch12(names_s[i])
# }
# match_res <- match(names_s, labels[,2])
# idx_y <- match_res[!is.na(match_res)]
# y <- as.integer(labels[idx_y, 1])
# idx_x <- which(!is.na(match_res))
# data <- array[,idx_x]

## approach 2
names_sample <- colnames(array)
fetch12 <- function(s) {
  paste0(unlist(strsplit(s,""))[1:12], collapse="")
}
selected <- NULL
y <- NULL
for (i in 1:ncol(array)) {
  temp <- fetch12(names_sample[i])
  # cat(i,temp,"\n")
  if (temp %in% labels[,2]) {
    selected <- c(selected, i)
    y <- c(y, as.integer(labels[which(labels[,2]==temp),1]))
  }
}

data <- array[,selected]

# load in network data
names_exp <- rownames(data)
elist <- as.matrix(read.table("network_data_converted_names.txt", header = T))
elist <- elist[-which(elist[,1]==elist[,2]),]
elist[,1] <- as.character(elist[,1])
elist[,2] <- as.character(elist[,2])
names_net <- unique(as.vector(elist))

## select network data
edgelist <- NULL
system.time(
for (i in 1:dim(elist)[1]) {
  if (elist[i,1]%in%names_exp & elist[i,2]%in%names_exp) {
    edgelist <- rbind(edgelist, elist[i,])
  } 
  if (i %% 100 == 0) {
    cat(i,"\n")
  }
}
)
length(unique(as.vector(edgelist)))
## talor the graph to contain only one connected component
library(igraph)
g <- graph.edgelist(edgelist, directed=F)
keepID <- which(components(g)$membership == 1)
names_net <- names(keepID)

## select expression data
X <- data[names_exp%in%names_net,]

## save the real gene names, re-naming genes
glist <- rownames(X)

G <- NULL
for (i in 1:dim(edgelist)[1]) {
  temp1 <- which(glist==edgelist[i,1])
  temp2 <- which(glist==edgelist[i,2])
  G <- rbind(G, c(temp1, temp2))
  if (i %% 100 == 0) {
    cat(i,"\n")
  }
}
G <- G - 1 ## for Python indices

# rownames(X) <- NULL
# colnames(X) <- NULL



## save G and X
write.csv(G, file="data_network.csv", row.names=F, col.names=F)
write.csv(cbind(t(X),y), file="data_expression.csv", row.names=F, col.names=F)
write.table(glist,file="GXglist.txt", row.names=F, col.names=F)

###################################################################################################

## make a toy graph
G <- as.matrix(read.csv("data_network.csv", header=F))
X <- as.matrix(read.csv("data_expression.csv", header=F))

# n_genes <- 1000
# n_samples <- 100
# 
# data_expression <- X[1:n_samples, 1:n_genes]
# data_network <- NULL
# for (i in 1:nrow(G)) {
#   if (G[i,1]<=n_genes & G[i,2]<=n_genes) {
#     data_network <- rbind(data_network, G[i,])
#   }
#   if (i %% 100 == 0) {
#     cat(i,"\n")
#   }
# }
# 
# library(igraph)
# g <- graph.edgelist(data_network, directed=F)
# hist(degree(g))
# 
# genes <- unique(as.vector(as.matrix(data_network)))
# length(genes)
# data_expression <- data_expression[1:n_samples, genes]
# 
# for (i in 1:nrow(data_network)) {
#   data_network[i,1] <- which(genes==data_network[i,1])
#   data_network[i,2] <- which(genes==data_network[i,2])
# }
# 
# write.csv(data_network, file="Tree_python/toy_network.csv", row.names=F, col.names=F)
# write.csv(data_expression, file="Tree_python/toy_expression.csv", row.names=F, col.names=F)

####################################################################################################

## analyze the network data

## make the final graph
library(igraph)
G <- G + 1
g <- graph.edgelist(G, directed=F)

## basic statistics
hist(degree(g),500)
dg <- degree(g)
components(g)$no ## the whole network is one connected component
diameter(g, directed=F) 

## make partition matrix
degree_threshold <- 15
length(dg[dg > degree_threshold])
centers <- V(g)[dg > degree_threshold]
length(centers)
partition <- matrix(0, nrow=length(V(g)), ncol=length(centers))
for (i in 1:ncol(partition)) {
  partition[c(G[which(G[,1]==centers[i]),2], G[which(G[,2]==centers[i]),1]),i] <- 1
}
n_neighbors <- apply(partition, 2, sum)
apply(partition, 1, sum) ## display overlapping/mixed membership
range(n_neighbors)
length(which(partition==1))

# write.table(partition, "partition.txt", row.names=F, col.names=F)
write.table(partition, "partition_sim.txt", row.names=F, col.names=F)

