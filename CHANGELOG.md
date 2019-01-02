# Changelog

## [0.1.6] - Jan 2, 2019

### Added

- Changelog to keep track of changes in each update.

### Changed

- Renamed `qc` module to `knn`, in accordance with the updated manuscript. This change clarifies the meaning of 'quality control' in this package, which is different from its typical meaning. In this package, quality control means to _explore the data according to certain qualities of samples and features_, rather than filtering the raw data according to technical parameters determined by domain specific knowledge.

- Renamed `qc.SampleKNNFilter` to `knn.RareSampleDetection`, in accordance with the updated manuscript. This change clarifies the purpose of this procedure, which is to identify samples distinct from their nearest neighbors _for further inspection rather than removal_. Filtering usually refers to the removal of samples.

- Renamed `qc.FeatureKNNPickUp` to `knn.FeatureImputation`, in accordance with the updated manuscript. This change clarifies the purpose of this procedure in the field of single-cell RNA-seq data analysis, which is to reasonably change zero entries in the data matrix into non-zero entries. This procedure is usually called 'imputation' in the field of single-cell RNA-seq data analysis.

- Moved `qc.remove_constant_features` to `utils.remove_constant_features`.
