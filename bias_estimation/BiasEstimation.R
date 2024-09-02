# BIAS estimation on testset
# This script assumes you have already extracted the embeddings for all images in the testset

library(readr)
library(data.table)
library(gridExtra)
library(matrixStats)
library(amap)
library(ggplot2)

# Load testset
load("~/Documents/Turing/ToPraveen/testset.RData")

# scaling function
colScale = function(x) {
  cm  = colMeans(x, na.rm = TRUE)
  csd = colSds(x, center = cm)
  x   = t( (t(x) - cm) / csd )
  return(x)
}

model <- 'FACENET'
path  <- paste0('/Users/santhoshnarayanan/Documents/Turing/SonyShades/storage/embeddings_FACENET.csv')
python_data    <- fread(path, header = FALSE, skip = 1)
python_data    <- python_data[,-1]

# Process subject id and filenames
python_data$V2 <- substr(python_data$V2, 70, 90)
python_data$id <- tstrsplit(python_data$V2, "/", keep = 1)
python_data$file <- substr(python_data$V2, 9, 20)
python_data <- python_data[,-1]

ncol        <- ncol(python_data)
python_data <- python_data[ , .SD , .SDcols = c(ncol-1, ncol, 1:(ncol-2))]
alldata     <- testset[python_data, on = c('id', 'file'), nomatch=NULL]
setkeyv(alldata, c('id', 'file'))
rm(list = c('python_data', 'path'))

play_data <- data.frame(alldata[,-c(1:4)])
dist_mat  <- as.matrix(Dist(play_data, nbproc = 15))
rm(play_data)

# Identification rates
niter <- 100
n_per_id <- 20
n_ids <- 800

results <- data.table()
for(i in 1:niter){
  set.seed(i)
  reference_set <- sample.int(n_per_id, n_ids, replace = TRUE)
  reference_set <- reference_set + n_per_id*seq(0, n_ids-1, 1)
  id_mat <- dist_mat[, reference_set]
  id_mat <- id_mat[-reference_set, ]
  id_set <- apply(id_mat, 1, FUN = which.min)
  ref_set <- rep(reference_set, each = n_per_id-1)
  res_dt <- data.table(alldata[-reference_set, c('skin_type', 'gender')], 
                       success = (alldata$id[ref_set] == alldata$id[reference_set][id_set]))
  res_dt <- res_dt[, list(num = .N, num_success = sum(success)), by = c('skin_type', 'gender')]
  setorder(res_dt, skin_type, gender) 
  if(ncol(results) > 0){
    results$num <- results$num + res_dt$num
    results$num_success <- results$num_success + res_dt$num_success
  }else{
    results <- cbind(results, res_dt)
  }
}
results$rate <- results$num_success/results$num
setorder(results, rate)
results$rate <- round(results$rate, 3)

pdf(paste0("FACENET_BIAS.pdf"), height=3, width=6)
grid.table(results)
dev.off()