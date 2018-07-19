# Used to simulate data for gene dropout correction
# benchmark
suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(splatter)
})


# Simulate Drop-seq data
# Macosko et al 2015 dataset obtained from
# https://hemberg-lab.github.io/scRNA.seq.datasets/mouse/retina/
ds_rds <- readRDS('macosko.rds')
# (n_genes, n_cells)
ds_cnt <- ds_rds@assays[['counts']]
set.seed(123)
ds_sample_inds <- sample(nrow(ds_cnt), 700)
ds_cnt_sample <- ds_cnt[, ds_sample_inds]
ds_params <- splatEstimate(ds_cnt_sample)
# change the number of genes to 20000
ds_params_20k_genes <- ds_params
ds_params_20k_genes@nGenes <- 20000
# view parameters
ds_params_20k_genes
# optionally save parameters for later usage
# saveRDS(ds_params_20k_genes, 'rds/ds_g20k_splatter_params.RDS')
# the number of samples to simulate
n_vec <- 500 * (2 ^ seq(from = 0, to = log2(20), length.out = 8))
plot(n_vec, 1:8, log='x')
# simulate
ds_sims <- lapply(n_vec, function(n) {
    seeds <- 1:5

    single_n_sims <- lapply(seeds, function(seed) {
        sim <- splatSimulateGroups(
            ds_params_20k_genes,
            batchCells      = as.integer(n),
            group.prob      = c(0.3, 0.2, 0.15, 0.15,  0.05, 0.05, 0.05, 0.05),
            de.prob         = c(0.2, 0.2, 0.25, 0.2 ,  0.2 , 0.2 , 0.2 , 0.2),
            de.facLoc       = 0,
            de.facScale     = c(1  , 0.9, 1   , 1   , 1.2 , 1.2 , 1.2 , 1.1),
            dropout.present = TRUE,
            dropout.mid     = -4.2,
            seed            = seed
        )
        return(sim)
    })
    return(single_n_sims)
})

# Simulate 10x genomics data
# PBMC68k dataset from
# https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a
# sample 1000 cells by the following Python code
#
# # transposed to (n_samples, n_features)
# np.random.seed(17)
# s1k_inds = np.random.choice(mat_arr.shape[0], 1000)
# 
# s1k_arr = mat_arr[s1k_inds]
# s1k_df = pd.DataFrame(s1k_arr)
# s1k_df.shape
# 
# s1k_df.to_csv('pbmc68k_s1k.csv')
#
# then, converted to csv
tenx_cnts <- read.csv('pbmc68k_s1k.csv')
# transpose to (n_features, n_samples)
tenx_params <- splatEstimate(t(tenx_x))
tenx_sims <- lapply(n_vec, function(n) {
    seeds <- 1:5

    single_n_sims <- lapply(seeds, function(seed) {
        sim <- splatSimulateGroups(
            tenx_params,
            batchCells      = as.integer(n),
            group.prob      = c(0.3, 0.2, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05),
            de.prob         = 0.12,
            de.facLoc       = 0,
            de.facScale     = 0.85,
            dropout.present = TRUE,
            dropout.mid     = -3.5,
            seed            = seed
        )
        return(sim)
    })
    return(single_n_sims)
})
