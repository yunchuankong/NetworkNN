setwd("C:\\Users\\kyccw\\Dropbox\\Research_Yu\\hurricane")
setwd("C:\\Users\\yunchuan\\Dropbox\\Research_Yu\\hurricane")
library(igraph)
library(mvtnorm)
library(Matrix)
seed = 1

# G <- as.matrix(read.csv("data_network.csv", header=F))
# g <- graph.edgelist(G, directed=F)
n_features <- 2000
n_samples <- 160
set.seed(seed)
g <- sample_pa(n_features, power=1/3, m=2, directed=F)
dg <- degree(g)
hist(dg,50)

# components(g)$no ## always 1
# vertex_connectivity(g) ## Do not calculate this for simulated graphs
transitivity(g, type="undirected") ## Clustering Coefficient
edge_density(g)
diameter(g, directed=F)
# eigen_centrality(g)
# plot(g, vertex.label=NA, vertex.size=4)

dist <- distances(g)
cov <- 0.75^dist
# for (i in 0:max(dist)) {
#   ind <- which(dist==i)
#   ind <- sample(ind, size=as.integer(length(ind)/2))
#   cov[ind] <- -cov[ind]
# }
diag(cov) <- 1
cov <- as.matrix(forceSymmetric(cov))


X_sim <- rmvnorm(n_samples, sigma = cov, method="svd")

cores <- which(dg>=18)[1:7] ## randomly choose 7 out of 13
length(cores)
predictors <- unique(unlist(adjacent_vertices(g, cores)))
length(predictors)

# Generate labels using logestic regression
beta0 <- runif(n=length(predictors)+1, 3, 5)
ind <- sample(1:length(beta0), size=as.integer(length(beta0)/3))
beta0[ind] <- -beta0[ind]
X0 <- X_sim[,predictors]
mu0 <- cbind(rep(1,nrow(X_sim)), X0)%*%beta0
logits0 <- exp(mu0)/(exp(mu0)+1)
y0 <- rep(0, nrow(X_sim))
y0[logits0>0.5] <- 1
sum(y0)/length(y0) ## proportion of labels 1

data_expression_sim <- cbind(X_sim,y0)
data_network_sim <- get.edgelist(g)
data_network_sim <- data_network_sim - 1 ## for Python

write.csv(data_expression_sim, file="data_expression_sim.csv", row.names=F, col.names=F)
## trivial to store data_network.csv for Neural: the partition will reduce genes involved in the network
# write.csv(data_network_sim, file="data_network_sim.csv", row.names=F, col.names=F)


### Create network partitions
G <- data_network_sim + 1
g <- graph.edgelist(G, directed=F)

## basic statistics
hist(degree(g),20)
dg <- degree(g)
components(g)$no ## the whole network is one connected component
diameter(g, directed=F) 

partition <- as.matrix(get.adjacency(g))


## make partition matrix
degree_threshold <- 8
length(dg[dg > degree_threshold])
centers <- V(g)[dg > degree_threshold]
length(centers)
partition <- matrix(0, nrow=length(V(g)), ncol=length(centers))
for (i in 1:ncol(partition)) {
  partition[c(G[which(G[,1]==centers[i]),2], G[which(G[,2]==centers[i]),1]),i] <- 1
}

n_neighbors <- apply(partition, 2, sum)
apply(partition, 1, sum) ## display overlapping/mixed membership
# omit <- which(apply(partition, 1, sum) == 0)
range(n_neighbors)
length(which(partition==1))

write.table(partition, "partition_sim.txt", row.names=F, col.names=F)
# write.table(partition[-omit,], "partition_sim.txt", row.names=F, col.names=F)
# write.csv(data_expression_sim[,-omit], file="data_expression_net_sim.csv", row.names=F, col.names=F)




