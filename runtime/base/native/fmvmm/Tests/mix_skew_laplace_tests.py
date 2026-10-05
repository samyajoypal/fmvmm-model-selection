# %% [markdown]
# # Example: Mixtures of Multivariate Skew Laplace Distributions
# %%
# import os
# os.chdir("../..")
# %%
# Let is first import the necessary modules

from fmvmm.mixtures.skewlaplacemix import SkewLaplaceMix

from sklearn import datasets
from fmvmm.utils.utils_mixture import clustering_metrics
# %%
# Let us load the data

iris = datasets.load_iris()

df = iris.data
# %%
mixskewlaplace = SkewLaplaceMix(3, print_log_likelihood= False, tol=1e-6, max_iter=100)
mixskewlaplace.fit(df)

# %%
# Get the fitted parapeters

print("pi: ", mixskewlaplace.get_params()[0])
print("alpha: ", mixskewlaplace.get_params()[1])
# %%
# Get Information Matrix and Standard Errors

im, se = mixskewlaplace.get_info_mat()

print("Standard Errors: ", se)
# %%
# Get AIC, BIC, ICL

print("AIC: ", mixskewlaplace.aic())
print("BIC: ", mixskewlaplace.bic())
print("ICL:", mixskewlaplace.icl())
# %%
# Check classification performance

clustering_metrics(iris.target, mixskewlaplace.predict())

# %%
