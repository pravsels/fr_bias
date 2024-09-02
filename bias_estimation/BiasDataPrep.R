# This script prepares a testset for Bias benchmarking
# This is based on the estimated skin luminosity and hue of all images

library(readr)
library(data.table)
library(gridExtra)
library(matrixStats)
library(amap)
library(ggplot2)

metadata <- fread("~/Documents/Turing/GPU/VGGData/VMER/identity_meta.csv")
colnames(metadata) <- c('id', 'name', 'count', 'flag', 'gender', 'age')

metadata$folder = paste0('/Users/santhoshnarayanan/Documents/Turing/GPU/VGGData/aligned/images/', metadata$id)
setorder(metadata, id)

# Import Skin Shade CSVs
setwd("/Users/santhoshnarayanan/Documents/Turing/SonyShades/storage/CSVs")
skin_shade_df <- do.call(rbind, lapply(list.files(pattern = "\\.csv$"), fread))
summary(skin_shade_df)

setwd("/Users/santhoshnarayanan/Documents/Turing/SonyShades/storage")

# Histograms
# hist(skin_shade_df$lum)
# hist(skin_shade_df$hue)
# hist(skin_shade_df$lum_std)
# hist(skin_shade_df$hue_std)

# Filtering Data
skin_shade_df <- skin_shade_df[skin_shade_df$hue > 0 & skin_shade_df$hue < 100]
skin_shade_df <- skin_shade_df[skin_shade_df$lum_std < 30]
skin_shade_df <- skin_shade_df[skin_shade_df$hue_std < 30]

skin_shade_df[, c('id') := tstrsplit(filename, '_', fixed = TRUE, keep = 1L)]
skin_shade_df[, c('file_1', 'file_2') := tstrsplit(filename, '_', fixed = TRUE, keep = c(2,3))]
skin_shade_df[, c('file') := paste0(file_1, '_', file_2, '.jpg')]
skin_shade_df[, c('file_1', 'file_2') := NULL]
skin_shade_df <- skin_shade_df[, c(6, 7, 2:5)]
length(unique(skin_shade_df$id))

round(table(skin_shade_df$lum < 60, skin_shade_df$hue > 55)/22941.55, 2)

# # Add density for each point
# skin_shade_df$density <- fields::interp.surface(
#   MASS::kde2d(skin_shade_df$hue, skin_shade_df$lum), 
#   skin_shade_df[,c("hue", "lum")])
# 
# # Plot
# ggplot(skin_shade_df, aes(hue, lum, alpha = 1/density)) +
#   geom_point(shape = 16, size = 1, show.legend = FALSE) +
#   theme_minimal() +
#   scale_color_gradient(low = "#32aeff", high = "#f2aeff") +
#   scale_alpha(range = c(.25, .6)) +
#   geom_vline(xintercept = 55, col = 'red', linetype = 2) +
#   geom_hline(yintercept = 60, col = 'red', linetype = 2)
# 
# skin_shade_df$density <- NULL

# Skin Shade Classes
quantile(skin_shade_df$lum, probs = seq(0, 1, 1/4))
quantile(skin_shade_df$hue, probs = seq(0, 1, 1/4))
skin_shade_df$lum_class <- cut(skin_shade_df$lum, breaks = c(0, 50, 70, 100), labels = c('dark', 'medium', 'light'))
skin_shade_df$hue_class <- cut(skin_shade_df$hue, breaks = c(0, 55, 100), labels = c('red', 'yellow'))
table(skin_shade_df$lum_class, skin_shade_df$hue_class)
skin_shade_df[, c('skin_type') := paste0(lum_class, '-', hue_class)]

# Benchmark Set Prep
set.seed(1234)

agg_df <- skin_shade_df[skin_type %in% c('light-red', 'light-yellow', 'dark-red', 'dark-yellow')]
agg_df <- agg_df[, list(freq = .N), by = c('id', 'skin_type')]

plot_df <- agg_df[, list(N_types = .N), by = id]
ggplot(plot_df) +
  geom_histogram(aes(x = N_types), binwidth=1) +
  theme_minimal()
rm(plot_df)

agg_df <- agg_df[freq >= 20]
agg_df <- metadata[, c('id', 'gender')][agg_df, on = 'id']
length(unique(agg_df$id))
agg_df$freq <- NULL
agg_df <- agg_df[, N := .N, by = id]
setorder(agg_df, N)
agg_df <- agg_df[,.SD[sample(.N)],by=N]
table(agg_df$N)
agg_df$N <- NULL
table(agg_df$skin_type)
#

# GREEDY ALGO
library(tidyverse)

# Getting unique IDs to iterate over
unique_ids <- unique(agg_df$id)

# Placeholder for final assignments
final_assignments <- data.frame(id = character(), skin_type = character())

# Track counts of each skin type assigned
skin_type_counts <- data.frame(skin_type = unique(agg_df$skin_type), 
                               count = integer(length(unique(agg_df$skin_type))), 
                               stringsAsFactors = FALSE)

for (id in unique_ids) {
  # Filter available options for this ID
  options <- filter(agg_df, id == !!id)
  
  if (nrow(options) == 1) {
    # If only one option, assign it directly
    final_assignments <- rbind(final_assignments, options)
    skin_type_counts$count[skin_type_counts$skin_type == options$skin_type] <- skin_type_counts$count[skin_type_counts$skin_type == options$skin_type] + 1
  } else {
    # Find the least represented skin_type that this ID can be assigned to
    possible_assignments <- merge(options, skin_type_counts, by = "skin_type")
    least_represented <- possible_assignments[which.min(possible_assignments$count),]
    
    final_assignments <- rbind(final_assignments, least_represented[1,-4])
    skin_type_counts$count[skin_type_counts$skin_type == least_represented$skin_type] <- 
      skin_type_counts$count[skin_type_counts$skin_type == least_represented$skin_type] + 1
  }
}

# View the distribution of skin types
print(skin_type_counts)
table(final_assignments$skin_type, final_assignments$gender)
####

set.seed(1234)
setorder(final_assignments, gender, skin_type, id)
final_assignments <- final_assignments[, .SD[sample(.N, 100, replace = FALSE)], 
                                       by = c('gender', 'skin_type')]

testset <- skin_shade_df[final_assignments, on = c('id', 'skin_type')]
setorder(testset, id, file)
testset <- testset[, .SD[sample(.N, 20, replace = FALSE)], by = c('id')]
table(testset$skin_type, testset$gender)

testset$source_path = paste0('/Users/santhoshnarayanan/Documents/Turing/GPU/VGGData/aligned/images/', 
                             testset$id, '/', testset$file)

testset$dest_path = paste0('/Users/santhoshnarayanan/Documents/Turing/SonyShades/storage/testset/', 
                           testset$id, '/', testset$file)

for(mdir in unique(testset$id)){
  
  mpath = paste0('/Users/santhoshnarayanan/Documents/Turing/SonyShades/storage/testset/', 
                 mdir)
  dir.create(mpath)
}

file.copy(from = testset$source_path, to = testset$dest_path)
testset <- testset[, c('id', 'file', 'gender', 'skin_type')]
rm(list = c('mdir', 'mpath', 'agg_df', 'least_represented',
            'options', 'possible_assignments', 'skin_type_counts', 'id', 'unique_ids'))
save.image("~/Documents/Turing/SonyShades/storage/Workspace.RData")
save(testset, file = "testset.RData")