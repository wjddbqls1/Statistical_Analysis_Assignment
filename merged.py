# ===== ch02.py =====
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[11]:


df = pd.read_csv ('data/ch1_sport_test.csv' ,
                  index_col= '학생번호' )
df


# In[5]:


jupyter nbconvert --to script 내파일.ipynb


# In[ ]:


get_ipython().system('jupyter nbconvert --to script 내파일.ipynb')



# ===== ch03.py =====
#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('precision', '3')


# In[44]:


df = pd.read_csv('../stat_analysis/data/ch2_scores_em.csv' ,
                  index_col = 'student number')
df.head()


# In[45]:


scores = np.array(df['english'])[:10]
scores


# In[46]:


scores_df = pd.DataFrame({'score':scores},
                         index=pd.Index(['A', 'B', 'C','D','E','F','G','H','I','J'],
                                        name = 'student'))
scores_df


# In[47]:


sum(scores) / len(scores)


# In[48]:


np.mean(scores)


# In[49]:


scores_df.mean()


# In[50]:


sorted_scores = np.sort(scores)
sorted_scores


# In[51]:


n = len (sorted_scores)
if n % 2 == 0:
    m0 = sorted_scores[n//2 -1]
    m1 = sorted_scores[n//2]
    median = (m0 + m1) /2
else :
    median + sorted_scores[(n+1)//2-1]
median


# In[52]:


np.median(scores)


# In[53]:


scores_df.median()


# In[54]:


pd.Series([1,1,1,2,2,3]).mode()


# In[56]:


pd.Series([1,2,3,4,5]).mode()


# In[57]:


mean = np.mean(scores)
deviation = scores - mean
deviation


# In[58]:


another_scores = [50, 60, 58, 54, 51, 56, 57, 53, 52, 59]
another_mean = np.mean(another_scores)
another_deviation = another_scores - another_mean
another_deviation


# In[59]:


np.mean(deviation)


# In[60]:


np.mean(another_deviation)


# In[61]:


summary_df = scores_df.copy()
summary_df['deviation']=deviation
summary_df


# In[62]:


summary_df.mean()


# In[63]:


np.mean(deviation ** 2)


# In[64]:


np.var(scores)


# In[65]:


scores_df.var()


# In[66]:


summary_df['square of deviation'] = np.square(deviation)
summary_df


# In[67]:


summary_df.mean()


# In[68]:


np.sqrt(np.var(scores, ddof=0))


# In[69]:


np.std(scores, ddof=0)


# In[70]:


np.max(scores) - np.min(scores)


# In[71]:


scores_Q1 = np.percentile(scores,25)
scores_Q3 = np.percentile(scores,75)
scores_IQR = scores_Q3 - scores_Q1
scores_IQR


# In[72]:


pd.Series(scores).describe()


# In[75]:


z=(scores - np.mean(scores)) / np.std(scores)
z


# In[76]:


np.mean(z),np.std(z, ddof=0)


# In[77]:


z = 50 + 10 * (scores - np.mean(scores)) / np.std(scores)
z


# In[79]:


scores_df['deviation value'] = z
scores_df


# In[81]:


english_scores = np.array(df['english'])
pd.Series(english_scores).describe()


# In[84]:


freq, _ = np.histogram(english_scores, bins=10, range=(0, 100))
freq


# In[85]:


freq_class = [f'{i}~{i+10}' for i in range (0, 100, 10)]
freq_dist_df = pd.DataFrame({'frequency':freq},
                            index=pd.Index(freq_class,
                                           name='class'))
freq_dist_df


# In[86]:


class_value = [(i+(i+10))//2 for i in range (0, 100, 10)]
class_value


# In[88]:


rel_freq = freq/freq.sum()
rel_freq


# In[89]:


cum_rel_freq = np.cumsum(rel_freq)
cum_rel_freq


# In[91]:


freq_dist_df['class value'] = class_value
freq_dist_df['relative frequency'] = rel_freq
freq_dist_df['cumulative relative frequency'] = cum_rel_freq
freq_dist_df = freq_dist_df[['class value', 'frequency', 'relative frequency', 'cumulative relative frequency']]
freq_dist_df


# In[93]:


freq_dist_df.loc[freq_dist_df['frequency'].idxmax(), 'class value']


# In[95]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[102]:


fig = plt.figure(figsize=(10,6))
ax=fig.add_subplot(111)
freq, _, _ = ax.hist(english_scores, bins=10, range=(0,100))
ax.set_xlabel('score')
ax.set_ylabel('person number')
ax.set_xticks(np.linspace(0,100,10+1))
ax.set_yticks(np.arange(0,freq.max()+1))
plt.show()


# In[104]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

freq, _ , _ = ax.hist(english_scores, bins=25, range=(0,100))
ax.set_xlabel('score')
ax.set_ylabel('person number')
ax.set_xticks(np.linspace(0,100,25+1))
ax.set_yticks(np.arange(0,freq.max()+1))
plt.show()


# In[106]:


fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

weights = np.ones_like(english_scores) / len(english_scores)
rel_freq, _, _ = ax1.hist(english_scores, bins=25,
                          range = (0, 100), weights=weights)
cum_rel_freq = np.cumsum(rel_freq)
class_value = [(i+(i+4))//2 for i in range (0, 100, 4)]
ax2.plot(class_value, cum_rel_freq,
         ls='--', marker='o', color='gray')
ax2.grid(visible=False)

ax1.set_xlabel('score')
ax1.set_ylabel('relative frequency')
ax2.set_ylabel('cumulative relative frequency')
ax1.set_xticks(np.linspace(0, 100, 25+1))

plt.show()


# In[ ]:





# In[ ]:






# ===== pg30.py =====
#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd


# In[10]:


df = pd.read_csv('../data/ch2_scores_em.csv' ,
                 index_col='student number')
df.head()


# In[12]:


scores = np.array(df['english'])[:10]
scores


# In[13]:


scores_df = pd.DataFrame({'score':scores},
                         index=pd.Index(['A','B','C','D','E',
                                         'F','G','H','I','J'],
                                        name='student'))
scores_df


# In[14]:


sum(scores)/len(scores)


# In[15]:


np.mean(scores)


# In[16]:


scores_df.mean()


# In[17]:


sorted_scores=np.sort(scores)
sorted_scores


# In[ ]:






