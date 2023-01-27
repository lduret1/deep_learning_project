# deep_learning_project
#design_parameters_tree.R : code to generate epidemiological parameters drawn from the prior distribution
#parameters_tree.csv : parameters drawn from the prior distributions
#simulation_and_encoding_tree.py : file to generate phylogenies using the prior distributions and encode them into summary statistics
#training set : contains 100 files, each file contains the values of summary statistics for 10^3 phylogenies
#rescaling_factors : contains 100 files, each files contains the values of rescaling factors for 10^3 phylogenies
#real_data : file containing the phylogenie and summary statistics for a HIV dataset
#abc.R : code to perform abc using training set and real_data as target
#ffnn.ipynb : code to train and test a neural network using training set and test_set
