#!/usr/bin/env python#!/usr/bin/python
# coding: utf-8

# In[2]:


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

# In[3]:


url = 'https://raw.githubusercontent.com'
url += '/CSSEGISandData/COVID-19'
url += '/master/csse_covid_19_data/csse_covid_19_time_series'
url += '/time_series_19-covid-Confirmed.csv'


# In[4]:


confirmed = pd.read_csv(url)


# In[5]:


confirmed.head()


# ### Preprocessing

# In[6]:


ger_confirmed = confirmed[confirmed['Country/Region']=='Germany'].T
ger_confirmed = ger_confirmed[4:].astype('int')
ger_confirmed.columns = ['confirmed']


# In[7]:


ger_confirmed.index = pd.to_datetime(ger_confirmed.index)
ger_confirmed = ger_confirmed.asfreq('D')


# Filter der Daten: Wir nehmen für die Modellbildung erst den Tag als Beginn, an dem der 100. Erkrankte gemeldet wurde.

# In[8]:


ger_confirmed = ger_confirmed[ger_confirmed.confirmed>=100]


# In[9]:


today = ger_confirmed.index[-1]


# ## Feature

# In[10]:


ger_confirmed['days'] = (ger_confirmed.index - ger_confirmed.index.min()).days


# In[11]:


ger_confirmed.head()


# ## Prediction Model

# In[12]:


from sklearn.linear_model import LinearRegression
import numpy as np


# In[13]:


X = ger_confirmed['days'].values.reshape(-1, 1)
y = ger_confirmed['confirmed'].values
logy = np.log(y)


# ### Train

# In[14]:


clf = LinearRegression()
clf.fit(X, logy)


# In[15]:


logy_pred = clf.predict(X)
ger_confirmed['predicted'] = np.exp(logy_pred).astype('int')


# In[16]:


ger_confirmed.tail()


# ## Save the model for later use

# In[17]:


import pickle

with open('%s-Germany-Covid19-Prediction-Model.pkl' % today.strftime('%Y-%m-%d'), 'wb') as f:
    pickle.dump(clf, f)


# ## Future

# In[18]:


fd = 13 # days into the future


# In[19]:


# Create DataFrame in the Future
dates = pd.date_range(ger_confirmed.index[-1], periods=fd, closed='right')
days_in_future = ger_confirmed.days[-1] + np.arange(1, fd)

future = pd.DataFrame(data=days_in_future, index=dates, columns=['days'])


# In[20]:


ger_future = ger_confirmed.append(future, sort=True)


# ### Predict the Future

# In[21]:


X_future = ger_future['days'].values.reshape(-1, 1)


# In[22]:


logy_pred = clf.predict(X_future)
ger_future['predicted'] = np.exp(logy_pred).astype('int')


# In[28]:


ger_future


# ## Future Plot

# In[23]:


title = 'Bestätigte Fälle und Vorhersage für Deutschland (Basierend auf CSSE COVID-19 Dataset)'


# In[24]:


ax = ger_future['confirmed'].plot(label='Bestätigte COVID-19 Fälle', marker='o')
ax = ger_future['predicted'].plot(label='exponentielles Wachstum\n(Modell vom %s)' % today.strftime('%d.%m.%Y'),
                                  alpha=0.6, ax=ax)

ax.legend()
ax.set_ylabel('Log(Anzahl)')
ax.set_yscale('log')
ax.set_title(title, fontsize=8)
ax.annotate('unter CC-BY 2.0 Lizenz Paul Balzer', xy=(.5, 0.02), xycoords='figure fraction', ha='center', fontsize=6, color='gray')

plt.tight_layout()
plt.savefig('./%s-Germany-Covid19-Prediction.png' % today.strftime('%Y-%m-%d'), dpi=150)


# ## Export as Excel

# In[25]:


ger_future.to_excel('./%s-Germany-Covid19-Prediction.xlsx' % today.strftime('%Y-%m-%d'))


# # Interactive Website
# 
# We are using Bokeh to export an interactive website

# In[26]:


from bokeh.plotting import figure
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models import Div, HoverTool, BoxAnnotation
from bokeh.layouts import column
from bokeh.io import output_file, save


# In[27]:


p = figure(tools="hover,save,pan,box_zoom,reset,wheel_zoom",
           x_axis_type="datetime", y_axis_type="log",
           title=title)

# Vorhersagemodell als Linie
p.line(ger_future.index, ger_future.predicted, line_width=5, line_color='darkred',
       legend='exponentielles Wachstum\n(Modell vom %s)' % today.strftime('%d.%m.%Y'))

# Tatsächliche Fälle als Punkte
p.circle(ger_confirmed.index, ger_confirmed.confirmed,
         fill_color="lightblue", size=12, legend='Bestätigte COVID-19 Fälle')

# X-Achse ordentlich formatieren
p.xaxis.formatter=DatetimeTickFormatter(
    years="%d.%m.%Y",
    months="%d.%m.%Y",
    days="%A %d.%m.%Y",
)

# Legende
p.legend.location = "top_left"

# Daten-Zeitraum
gray_box = BoxAnnotation(left=ger_confirmed.index[0],
                          right=ger_confirmed.index[-1],
                          fill_color='gray', fill_alpha=0.1)
p.add_layout(gray_box)

# Tooltips
p.select_one(HoverTool).tooltips = [
    ('Datum', '@x{%d.%m.%Y}'),
    ('Fälle', '@y{0.0a}'),
]
p.select_one(HoverTool).formatters = {'x':'datetime'}
p.select_one(HoverTool).mode = 'vline'

# Anmerkung
div = Div(text="""Quellcode: <a href="https://github.com/balzer82/covid-germany-predictor">Covid Germany Predictor</a> unter CC-BY2.0 Lizenz on Github.""")

# Save
output_file("index.html", title='Covid19 Prediction Germany')

save(column(p, div, sizing_mode="stretch_both"))


# CC-BY 2.0 Paul Balzer
