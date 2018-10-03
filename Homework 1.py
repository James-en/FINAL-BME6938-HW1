
# coding: utf-8

# # Homework 1

# ## Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sb


# ## Part A

# In[2]:


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/postoperative-patient-data/post-operative.data"
titles = ('L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT', 'decision ADM-DECS')
data = pd.read_csv(url,names=titles)


# ## Part C (Pre-Processing)
# ### Line 1 deletes all rows that have a '?' in the COMFORT column
# ### Line 2 removes the space after the A in the last column
# ### Line 3 converts the strings in the COMFORT column to their numeric value

# In[3]:


fixeddata = data.replace({'?':np.nan}).dropna()
fixeddata = fixeddata.replace({'A ':'A'})
fixeddata["COMFORT"] = pd.to_numeric(fixeddata["COMFORT"])


# ## Part B (Plots)
# ## L-Core vs. COMFORT

# In[4]:


graph = sb.swarmplot(x=fixeddata['L-CORE'], y=fixeddata['COMFORT'], data=fixeddata, order=["low", "mid", "high"])


# ## L-SURF vs. COMFORT

# In[5]:


graph = sb.swarmplot(x=fixeddata['L-SURF'], y=fixeddata['COMFORT'], data=fixeddata, order=["low", "mid", "high"])


# ## L-O2 vs. COMFORT

# In[6]:


graph = sb.swarmplot(x=fixeddata['L-O2'], y=fixeddata['COMFORT'], data=fixeddata, order=["poor", "fair", "good", "excellent"])


# ## L-BP vs. COMFORT

# In[7]:


graph = sb.swarmplot(x=fixeddata['L-BP'], y=fixeddata['COMFORT'], data=fixeddata, order=["low", "mid", "high"])


# ## SURF-STBL vs. COMFORT

# In[8]:


graph = sb.swarmplot(x=fixeddata['SURF-STBL'], y=fixeddata['COMFORT'], data=fixeddata, order=["unstable", "mod-stable", "stable"])


# ## CORE-STBL vs. COMFORT

# In[9]:


graph = sb.swarmplot(x=fixeddata['CORE-STBL'], y=fixeddata['COMFORT'], data=fixeddata, order=["unstable", "mod-stable", "stable"])


# ## BP-STBL vs. COMFORT

# In[10]:


graph = sb.swarmplot(x=fixeddata['BP-STBL'], y=fixeddata['COMFORT'], data=fixeddata, order=["unstable", "mod-stable", "stable"])


# ## Decision ADM-DECS vs. COMFORT

# In[11]:


graph = sb.swarmplot(x=fixeddata['decision ADM-DECS'], y=fixeddata['COMFORT'], data=fixeddata, order=["A", "I", "S"])


# ## Interpretation of plots
# ### Based on the 8 plots above, the categorical feature that has the most direct correlation with high COMFORT scores is the CORE-STBL feature. A large portion of the 15's reported in the COMFORT category came from patients that were reported as 'stable' in the CORE-STBL feature.
