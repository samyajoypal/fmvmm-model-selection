# %% [markdown]
# # Example: FMVMM (Mixtures of Identical and Non-Identical Distributions)
# %%
# import os
# os.chdir("../..")
# %%
# Let us first import necessary modules

from fmvmm.mixtures.FMVMM import fmvmm
from sklearn import datasets
from fmvmm.utils.utils_mixture import clustering_metrics 
# %%
# LEt us first get iris data

iris = datasets.load_iris()

df = iris.data
# %%
# We can  fit mixtures of all possible combinations of identical and non-identical distributions.
#However, that might take quite an amount of time.

# Let us try all possible combinations of the following distributions: 
# Multivariate Generalized Hyperbolic,
# Skew Normal
# Multivariate Normal Inverse Gaussian
# Multivariate T
# Multivariate Skew T

dist = ["mghp", "mvsn", "mnig","mvt","mvst"]

#If you want to use the list of all the distributions do not provide list_of_dist argument.
#By default it will use the full list of distributions

model = fmvmm(n_clusters=3, list_of_dist = dist, debug=True)

model.fit(df)
# %%
# Find the best model based on bic

model.best_mixture()
# %%
# Get BIC of all the fitted models

model.bic()
# %%
# get information matrix and standard errors for all the fitted models

ims, ses = model.get_info_mat()
# %%
# get weighted information matrix and standard errors for all the fitted models

ims_w, ses_w = model.get_info_mat_soft()
# %%
# We can see Out of the fitted models which were successfull

model.worked()
# %%
# We can see Out of the fitted models which were not successfull

model.not_worked()
# %%
# Luckily all models were successfull.

# We can get the MLEs for all the models

pi_list, alpha_list = model.get_params()
# %%
# let us now get the MLEs of the best model

pi_best, alpha_best = model.best_params()

print("pi: ", pi_best)
print("alpha: ", alpha_best)
# %%
# Check the clustering performance

from fmvmm.utils.utils_mixture import clustering_metrics

# To get the predictions for all the models

cluster_list = model.predict()

# To get the predictions for the best model

best_cluster = model.best_predict()

clustering_metrics(iris.target, best_cluster)
# %%
# Get the top performing models with the BIC values

model.get_top_mixtures()
# %%
print(ses[0])
# %%
print(ses_w[0])
# %%
