#!/usr/bin/env python#!/usr/bin/python
# coding: utf-8

# In[74]:


import pandas as pd


import matplotlib.pyplot as plt
plt.style.use('bmh')


# # WirVsVirus Hackathon
# 
# Die entscheidende Frage bei der Beurteilung aller Maßnahmen ist, ob das exponentielle Wachstum verlangsamt worden ist, d.h. die exponentielle Wachstumskurve abflacht.
# Dazu macht man am besten anhand bestehender Daten ein Modell-Fit und schaut, ob aktuelle Fallzahlen das Modell überschreiten oder man mit den Fallzahlen darunter bleibt.

# ## Download Data from CSSE COVID-19 Dataset
# 
# We are using the Covid-19 Dataset: https://github.com/CSSEGISandData/COVID-19

# In[75]:


url = 'https://raw.githubusercontent.com'
url += '/CSSEGISandData/COVID-19'
url += '/master/csse_covid_19_data/csse_covid_19_time_series'
url += '/time_series_19-covid-Deaths.csv'


# In[76]:


deaths = pd.read_csv(url)


# In[77]:


deaths.head()


# ### Preprocessing

# In[78]:


ger_deaths = deaths[deaths['Country/Region']=='Germany'].T
ger_deaths = ger_deaths[4:].astype('int')
ger_deaths.columns = ['deaths']


# In[79]:


ger_deaths.index = pd.to_datetime(ger_deaths.index)
ger_deaths = ger_deaths.asfreq('D')


# Filter der Daten: Wir nehmen für die Modellbildung erst den Tag als Beginn, an dem der 10. Tote gemeldet wurde.

# In[80]:


ger_deaths = ger_deaths[ger_deaths.deaths >= 10]


# In[81]:


today = ger_deaths.index[-1]


# ## Feature

# In[82]:


ger_deaths['days'] = (ger_deaths.index - ger_deaths.index.min()).days


# In[83]:


ger_deaths.head()


# ## Prediction Model

# In[84]:


from sklearn.linear_model import LinearRegression
import numpy as np


# In[85]:


X = ger_deaths['days'].values.reshape(-1, 1)
y = ger_deaths['deaths'].values
logy = np.log(y)


# ### Train

# In[86]:


clf = LinearRegression()
clf.fit(X, logy)


# In[87]:


logy_pred = clf.predict(X)
ger_deaths['predicted'] = np.exp(logy_pred).astype('int')


# In[88]:


ger_deaths.tail()


# ## Future

# In[89]:


fd = 13 # days into the future


# In[90]:


# Create DataFrame in the Future
dates = pd.date_range(ger_deaths.index[-1], periods=fd, closed='right')
days_in_future = ger_deaths.days[-1] + np.arange(1, fd)

future = pd.DataFrame(data=days_in_future, index=dates, columns=['days'])


# In[91]:


ger_future = ger_deaths.append(future, sort=True)


# ### Predict the Future

# In[92]:


X_future = ger_future['days'].values.reshape(-1, 1)


# In[93]:


logy_pred = clf.predict(X_future)
ger_future['predicted'] = np.exp(logy_pred).astype('int')


# In[94]:


ger_future


# ## Future Plot

# In[95]:


title = 'Todesfälle und Vorhersage für Deutschland (Basierend auf CSSE COVID-19 Dataset)'


# In[96]:


ax = ger_future['deaths'].plot(label='Bestätigte COVID-19 Tote', marker='o')
ax = ger_future['predicted'].plot(label='exponentielles Wachstum\n(Modell vom %s)' % today.strftime('%d.%m.%Y'),
                                  alpha=0.6, ax=ax)

ax.legend()
ax.set_ylabel('Log(Anzahl)')
ax.set_yscale('log')
ax.set_title(title, fontsize=8)
ax.annotate('unter CC-BY 2.0 Lizenz Paul Balzer', xy=(.5, 0.02), xycoords='figure fraction', ha='center', fontsize=6, color='gray')

plt.tight_layout()
plt.savefig('./%s-Germany-Covid19-Death-Prediction.png' % today.strftime('%Y-%m-%d'), dpi=150)


# ## Export as Excel

# In[97]:


ger_future.to_excel('./%s-Germany-Covid19-Death-Prediction.xlsx' % today.strftime('%Y-%m-%d'))


# CC-BY 2.0 Paul Balzer
