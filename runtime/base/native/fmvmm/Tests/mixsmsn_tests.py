# %% [markdown]
# # Example: Mixtures of Scale Mixtures of Skew Normal Distributions
# %%
# import os
# os.chdir("../..")
# %%
# Let is first import the necessary modules

from fmvmm.mixtures.skewtmix_smsn import SkewTMix
from fmvmm.mixtures.tmix_smsn import TMix
from fmvmm.mixtures.skewcontmix_smsn import SkewContMix
from fmvmm.mixtures.skewslashmix_smsn import SkewSlashMix
from fmvmm.mixtures.slashmix_smsn import SlashMix
from fmvmm.mixtures.skewnormmix_smsn import SkewNormalMix

from sklearn import datasets
from fmvmm.utils.utils_mixture import clustering_metrics
# %%
# Let us load the data

iris = datasets.load_iris()

df = iris.data
# %%
# Fit Skew T Mixtures

mixskt = SkewTMix(3, print_log_likelihood= True)
mixskt.fit(df)
# %%
# Get the fitted parapeters

print("pi: ", mixskt.get_params()[0])
print("alpha: ", mixskt.get_params()[1])
# %%
# Get Information Matrix and Standard Errors

im, se = mixskt.get_info_mat()

print("Standard Errors: ", se)
# %%
# Get AIC, BIC, ICL

print("AIC: ", mixskt.aic())
print("BIC: ", mixskt.bic())
print("ICL:", mixskt.icl())
# %%
# Check classification performance

clustering_metrics(iris.target, mixskt.predict())
# %%
# Fit T Mixtures

mixt = TMix(3, print_log_likelihood= True)
mixt.fit(df)

# %%
# Get the fitted parapeters

print("pi: ", mixt.get_params()[0])
print("alpha: ", mixt.get_params()[1])
# %%
# Get Information Matrix and Standard Errors

im, se = mixt.get_info_mat()

print("Standard Errors: ", se)
# %%
# Get AIC, BIC, ICL

print("AIC: ", mixt.aic())
print("BIC: ", mixt.bic())
print("ICL:", mixt.icl())
# %%
# Check classification performance

clustering_metrics(iris.target, mixt.predict())
# %%
# Fit Skew Normal Contaminated Mixtures

mixcont = SkewContMix(3, print_log_likelihood= True)
mixcont.fit(df)

# %%
# Get the fitted parapeters

print("pi: ", mixcont.get_params()[0])
print("alpha: ", mixcont.get_params()[1])
# %%
# Get Information Matrix and Standard Errors

im, se = mixcont.get_info_mat()

print("Standard Errors: ", se)
# %%
# Get AIC, BIC, ICL

print("AIC: ", mixcont.aic())
print("BIC: ", mixcont.bic())
print("ICL:", mixcont.icl())
# %%
# Check classification performance

clustering_metrics(iris.target, mixcont.predict())
# %%
# Fit Skew Slash Mixtures

mixslash = SkewSlashMix(3, print_log_likelihood= True)
mixslash.fit(df)

# %%
# Get the fitted parapeters

print("pi: ", mixslash.get_params()[0])
print("alpha: ", mixslash.get_params()[1])
# %%
# Get Information Matrix and Standard Errors

im, se = mixslash.get_info_mat()

print("Standard Errors: ", se)
# %%
# Get AIC, BIC, ICL

print("AIC: ", mixslash.aic())
print("BIC: ", mixslash.bic())
print("ICL:", mixslash.icl())
# %%
# Check classification performance

clustering_metrics(iris.target, mixslash.predict())
# %%
# Fit Slash Mixture

mixslash_s = SlashMix(3, print_log_likelihood= True)
mixslash_s.fit(df)

# %%
# Get the fitted parapeters

print("pi: ", mixslash_s.get_params()[0])
print("alpha: ", mixslash_s.get_params()[1])
# %%
# Get Information Matrix and Standard Errors

im, se = mixslash_s.get_info_mat()

print("Standard Errors: ", se)
# %%
# Get AIC, BIC, ICL

print("AIC: ", mixslash_s.aic())
print("BIC: ", mixslash_s.bic())
print("ICL:", mixslash_s.icl())
# %%
# Check classification performance

clustering_metrics(iris.target, mixslash_s.predict())
# %%
# Fit Skew Normal Mix

mixskewnorm = SkewNormalMix(3, print_log_likelihood= True, tol=1e-6, max_iter=25)
mixskewnorm.fit(df)

# %%
# Get the fitted parapeters

print("pi: ", mixskewnorm.get_params()[0])
print("alpha: ", mixskewnorm.get_params()[1])
# %%
# Get Information Matrix and Standard Errors

im, se = mixskewnorm.get_info_mat()

print("Standard Errors: ", se)
# %%
# Get AIC, BIC, ICL

print("AIC: ", mixskewnorm.aic())
print("BIC: ", mixskewnorm.bic())
print("ICL:", mixskewnorm.icl())
# %%
# Check classification performance

clustering_metrics(iris.target, mixskewnorm.predict())
# %%
