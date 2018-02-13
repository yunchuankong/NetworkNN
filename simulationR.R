
# setwd("C:\\Users\\kyccw\\Dropbox\\Research_Yu\\hurricane")
setwd("C:\\Users\\yunchuan\\Dropbox\\Research_Yu\\jungle")
# setwd("C:/Users/yunchuan/Documents/hurricane_artificial_data")
# library(igraph)
# library(mvtnorm)
# library(Matrix)
# seed = 1

generate_data <- function(nums){

  n_cores <- nums[1]
  singleton_prop <- nums[2]
  beta_lb <- nums[3]
  beta_ub <- nums[4]

  n_features <- 5000
  n_samples <- 400
  # set.seed(seed)
  g <- sample_pa(n_features, power=0.8, m=2, directed=F)
  dg <- degree(g)

  dist <- distances(g)
  cov <- 0.6^dist
  diag(cov) <- 1
  cov <- as.matrix(forceSymmetric(cov))
  # set.seed(seed)
  X_sim <- rmvnorm(n_samples, sigma = cov, method="svd")

  cores <- sample(which(dg>=20), n_cores)
  predictors <- c(cores, unique(unlist(adjacent_vertices(g, cores))))[1:(n_cores*20)]
  n_singleton <- ceiling(length(predictors)*singleton_prop)
  singleton_ind <- sample(length(predictors), n_singleton)
  predictors[singleton_ind] <- sample((1:n_features)[-predictors], n_singleton)

  # set.seed(seed)
  beta0 <- runif(n=length(predictors)+1, beta_lb, beta_ub)
  # beta0 <- rep(1.5,length(predictors)+1)
  neg_ind <- sample(1:length(beta0), size=as.integer(length(beta0)/3)) ## let some betas be negative
  beta0[neg_ind] <- -beta0[neg_ind]
  X0 <- X_sim[,predictors]
  mu0 <- cbind(rep(1,nrow(X_sim)), X0)%*%beta0
  logistic0 <- exp(mu0)/(exp(mu0)+1)
  
  # pdf("temp.pdf")
  # plot(density(logistic0))
  # abline(v=logistic0[tail(order(logistic0), n_samples*160/400)[1]])
  # dev.off()
  
  y0 <- rep(0, nrow(X_sim))
  logistic0_selected <- tail(order(logistic0), n_samples*160/400)
  y0[logistic0_selected] <- 1

  data_expression_sim <- cbind(X_sim,y0)
  write.csv(data_expression_sim, file="data_expression_sim.csv", row.names=F, col.names=F)

  # data_network_sim <- get.edgelist(g)
  # data_network_sim <- data_network_sim - 1 ## for Python
  # write.csv(data_network_sim, file="data_network_sim.csv", row.names=F, col.names=F)

  partition <- as.matrix(get.adjacency(g))
  diag(partition) <- 1
  write.table(partition, "partition_sim.txt", row.names=F, col.names=F)
}

run_DFN <- function(file){
  command = "python"
  path2script="C:/Users/yunchuan/Dropbox/Research_Yu/jungle/nn_call.py"
  allArgs = c(path2script, file)
  output = system2(command, args=allArgs, stdout=TRUE)
  return(as.numeric(output)[1]) 
}

run_GEDFN <- function(file){
  command = "python"
  path2script="C:/Users/yunchuan/Dropbox/Research_Yu/jungle/network_nn_call.py"
  allArgs = c(path2script, file)
  output = system2(command, args=allArgs, stdout=TRUE)
  # print(output)
  return(as.numeric(output)[1]) 
}

run_NGF <- function(file){
  command = "python"
  path2script="C:/Users/yunchuan/Dropbox/Research_Yu/jungle/simulation_ngf/network_forest_call.py"
  allArgs = c(path2script, file)
  output = system2(command, args=allArgs, stdout=TRUE)
  return(as.numeric(output[1])) 
}

nc_max <- 5
times <- 10
# sing0 <- matrix(0, nrow=3, ncol=nc_max)
# sing01 <- matrix(0, nrow=3, ncol=nc_max)
sing1 <- matrix(0, nrow=3, ncol=nc_max)
for (nc in 1:nc_max){
  dfn <- 0
  gedfn <- 0
  ngf <- 0
  for (i in 1:times){
    ## dataset
    # nums <- c(2*nc, 1, 0, 0.1)
    # generate_data(nums)
	filename <- paste0("core", nc, "sing3","num", i+10)
    path <- paste0("C:/Users/yunchuan/Documents/hurricane_artificial_data/", filename) ## sing1,2,3
    file <- c(paste0(path,"_expression.csv"),
             paste0(path,"_adjacency.txt"),
             paste0(path,"_edgelist.csv"))
    ## DFN
    temp_dfn <- run_DFN(file)
    dfn <- temp_dfn/times + dfn
    
    ## GEDFN
    temp_gedfn <- run_GEDFN(file)
    gedfn <- temp_gedfn/times + gedfn
    # temp_gedfn <- 0 ## temp use, comment out it when activating GEDFN
    
    ## NGF
    temp_ngf <- run_NGF(file)
    ngf <- temp_ngf/times + ngf
    # temp_ngf <- 0 ## temp use, comment out it when activating NGF
    
    ## log
    cat(filename, "auc:", temp_dfn, temp_gedfn, temp_ngf, "\n")
  }
  # sing0[1, nc] <- dfn
  # sing0[2, nc] <- gedfn
  # sing0[3, nc] <- ngf
  # write.table(sing0, "results_sing0.txt", row.names=F, col.names=F)
  
  # sing01[1, nc] <- dfn
  # sing01[2, nc] <- gedfn
  # sing01[3, nc] <- ngf
  # write.table(sing01, "results_sing01.txt", row.names=F, col.names=F)
  
  sing1[1, nc] <- dfn
  sing1[2, nc] <- gedfn
  sing1[3, nc] <- ngf
  write.table(sing1, "results_sing1.txt", row.names=F, col.names=F)
  
}

# write.csv(round(rbind(sing0,sing01,sing1),3),file="results_11-20_300trees.csv",col.names=F,row.names=F)