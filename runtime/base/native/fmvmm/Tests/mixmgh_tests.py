# %% [markdown]
# # Example: Mixtures of Multivariate Generalized Hyperbolic Distributions
# %%
# import os
# os.chdir("../..")
# %%
# Let is first import the necessary modules

from fmvmm.mixtures.mixmgh import MixMGH

from sklearn import datasets
from fmvmm.utils.utils_mixture import clustering_metrics
# %%
# Let us load the data

iris = datasets.load_iris()

df = iris.data
# %%
mixmgh = MixMGH(3)
mixmgh.fit(df)
# %%
# Get the fitted parapeters

print("pi: ", mixmgh.get_params()[0])
print("alpha: ", mixmgh.get_params()[1])
# %%
# Get Information Matrix and Standard Errors

im, se = mixmgh.get_info_mat()

print("Standard Errors: ", se)
# %%
# Get AIC, BIC, ICL

print("AIC: ", mixmgh.aic())
print("BIC: ", mixmgh.bic())
print("ICL:", mixmgh.icl())
# %%
# Check classification performance

clustering_metrics(iris.target, mixmgh.predict())

# %%
