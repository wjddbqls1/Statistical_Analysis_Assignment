#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('precision', '3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


x_set = np.array([1,2,3,4,5,6])


# In[42]:


def f(x) :
    if x in x_set:
        return x / 21
    else:
        return 0


# In[93]:


X = [x_set, f]


# In[94]:


prob = np.array([f(x_k) for x_k in x_set])

dict(zip(x_set, prob))


# In[95]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.bar(x_set, prob)
ax.set_xlabel('value')
ax.set_ylabel('probability')

plt.show()


# In[96]:


a = {'사과':1 , '딸기':5, '귤':10}


# In[97]:


a


# In[98]:


a = {('초콜릿', 200):20, ('마카롱', 500):15, ('쿠키', 300):30}
a


# In[99]:


a = {'사과':1, '딸기':5, '귤':10}
v1 = a['딸기']
v1


# In[100]:


v2 = a ['레몬']
v2


# In[101]:


f1 = '딸기' in a
f1


# In[102]:


f2 = '레몬' not in a
f2


# In[103]:


f3 = '레몬' in a
f3


# In[104]:


v1 = a.get('딸기')
v1


# In[105]:


v2 = a.get('레몬')
v2


# In[106]:


a = {'초콜릿':1, '마카롱':2, '쿠키':3}
a['초콜릿'] = 'One'
a['마카롱'] = 'Two'
a['쿠키'] = 'Three'
a


# In[107]:


d = dict (초콜릿 = 20, 마카롱 = 15, 쿠키 = 30)
d


# In[108]:


key = ['초콜릿', '마카롱', '쿠키']
value = [20, 15, 30]
d = dict(zip(key, value))
d


# In[109]:


d = dict ([('초콜릿', 20), ('마카롱', 15), ('쿠키', 30)])
d


# In[110]:


np.all(prob >= 0)


# In[111]:


np.sum(prob)


# In[112]:


def F(x):
    return np.sum([f(x_k) for x_k in x_set if x_k <= x])


# In[113]:


F(3)


# In[114]:


y_set = np.array([2 * x_k + 3 for x_k in x_set])
prob = np.array([f(x_k) for x_k in x_set])
dict(zip(y_set, prob))


# In[115]:


np.sum([x_k * f(x_k) for x_k in x_set])


# In[116]:


np.random.choice(5, 5, replace=False)


# In[117]:


np.random.choice(5, 3, replace=False)


# In[118]:


np.random.choice(5, 10)


# In[119]:


np.random.choice(5,10, p=[0.1, 0, 0.3, 0.6, 0])


# In[120]:


sample = np.random.choice(x_set, int(1e6), p=prob)
np.mean(sample)


# In[121]:


def E(X, g=lambda x: x):
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])


# In[123]:


E(X)


# In[124]:


E(X, g=lambda x: 2*x +3)


# In[125]:


2 * E(X) + 3


# In[126]:


strings = ['hyeja', 'parkhyeja', 'youngtae', 'kimyoungtae', 'bbangtae']


# In[127]:


strings.sort(key=lambda x: len(set(list(x))))


# In[128]:


strings


# In[129]:


2 * E(X) + 3


# In[131]:


mean = E(X)
np.sum([(x_k-mean)**2 * f(x_k) for x_k in x_set])


# In[133]:


def V(X, g=lambda x: x):
    x_set, f = X
    mean = E(X, g)
    return np.sum([(g(x_k)-mean)**2 * f(x_k) for x_k in x_set])


# In[134]:


V(X)


# In[136]:


V(X, lambda x: 2*x + 3)


# In[139]:


2**2 * V(X)


# In[140]:


get_ipython().system('jupyter nbconvert --to script "7주차.ipynb"')


# In[ ]:




