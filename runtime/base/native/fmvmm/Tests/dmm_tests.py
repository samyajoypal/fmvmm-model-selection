# %% [markdown]
# # Example: Dirichlet Mixture Model
# %%
# import os
# os.chdir("../..")
# %%
# import necessary modules

from fmvmm.mixtures.DMM_Soft import DMM_Soft
from fmvmm.mixtures.DMM_Hard import DMM_Hard
from fmvmm.utils.utils_mixture import sample_mixture_distribution
from fmvmm.distributions import dirichlet
import numpy as np
from fmvmm.utils.utils_dmm import wald_confidence_intervals_dmm
# %%
# Let us first generate some data from a dirichlet mixture model
np.random.seed(5)
pis = [0.3,0.5,0.2]
a1 =[5,5,5]
a2 = [7,15,225]
a3 = [50,10,4]
alphas = [[a1],[a2],[a3]]

data, label = sample_mixture_distribution(1000, dirichlet.rvs, pis,alphas)
# %%
# First Let us fit Soft DMM to the Data

model1 = DMM_Soft(n_clusters= 3)
model1.fit(data)
# %%
# To get the MLE of the parameters:

pi_soft, alpha_soft = model1.get_params()

print("pi vaues: ", pi_soft)
print("alpha vaues: ", alpha_soft)
# %%
# To get the standard errors:

im, se = model1.get_info_mat(method="louis")

print("standard errors: ", se)
# %%
# Confidence Intervals
ci = wald_confidence_intervals_dmm(model1.get_params(),im, alpha=0.05)
for i, (lo, hi) in enumerate(ci):
    print(f"Param {i+1}: ({lo:.4f}, {hi:.4f})")
# %%
# To get the standard errors:

im, se = model1.get_info_mat(method="score")

print("standard errors: ", se)
# %%
# Confidence Intervals
ci = wald_confidence_intervals_dmm(model1.get_params(),im, alpha=0.05)
for i, (lo, hi) in enumerate(ci):
    print(f"Param {i+1}: ({lo:.4f}, {hi:.4f})")
# %%
# To check classification performance keeping in mind label switching:

from fmvmm.utils.utils_mixture import clustering_metrics

clustering_metrics(label,model1.predict())
# %%
# To get AIC, BIC, ICL

print("AIC", model1.aic())
print("BIC", model1.bic())
print("ICL", model1.icl())
# %%
# We can similarly fit Hard DMM

model2 = DMM_Hard(n_clusters= 3)
model2.fit(data)
# %%
# To get the MLE of the parameters:

pi_soft, alpha_soft = model2.get_params()

print("pi vaues: ", pi_soft)
print("alpha vaues: ", alpha_soft)
# %%
# To get the standard errors:

im, se = model2.get_info_mat(method="louis")

print("standard errors: ", se)
# %%
# Confidence Intervals
ci = wald_confidence_intervals_dmm(model2.get_params(),im, alpha=0.05)
for i, (lo, hi) in enumerate(ci):
    print(f"Param {i+1}: ({lo:.4f}, {hi:.4f})")
# %%
# To get the standard errors:

im, se = model2.get_info_mat(method="score")

print("standard errors: ", se)
# %%
# Confidence Intervals
ci = wald_confidence_intervals_dmm(model2.get_params(),im, alpha=0.05)
for i, (lo, hi) in enumerate(ci):
    print(f"Param {i+1}: ({lo:.4f}, {hi:.4f})")
# %%
# To check classification performance keeping in mind label switching:


clustering_metrics(label,model2.predict())
# %%
# To get AIC, BIC, ICL

print("AIC", model2.aic())
print("BIC", model2.bic())
print("ICL", model2.icl())
# %% [markdown]
# ## High Dimensional Case
# 
# Only for Soft DMM it is currently implemented 
# %%
import numpy as np

pis = [0.4762, 0.2857, 0.2381]

a1=np.random.uniform(10,20,1000)
a2=np.random.uniform(20,200,1000)
a3=np.random.uniform(10,100,1000)

alphas = [[a1],[a2],[a3]]

data, label = sample_mixture_distribution(1000, dirichlet.rvs, pis,alphas)
# %%
# Method: Highdimensional 

model3 = DMM_Soft(n_clusters= 3, method="highdimensional")
model3.fit(data)
# %%
# To check classification performance 

clustering_metrics(label,model3.predict())
# %%
#Execution Time in Seconds

model3.execution_time
# %%
