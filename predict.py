#!/usr/bin/env python#!/usr/bin/python
# coding: utf-8

# In[1]:


import pandas as pd
import time
import datetime


import matplotlib.pyplot as plt
plt.style.use('bmh')


# # WirVsVirus Hackathon
# 
# Die entscheidende Frage bei der Beurteilung aller Maßnahmen ist, ob das exponentielle Wachstum verlangsamt worden ist, d.h. die exponentielle Wachstumskurve abflacht.
# Dazu macht man am besten anhand bestehender Daten ein Modell-Fit und schaut, ob aktuelle Fallzahlen das Modell überschreiten oder man mit den Fallzahlen darunter bleibt.

# ## Download Data from CSSE COVID-19 Dataset
# 
# We are using the Covid-19 Dataset: https://github.com/CSSEGISandData/COVID-19

# In[2]:


url = 'https://raw.githubusercontent.com'
url += '/CSSEGISandData/COVID-19'
url += '/master/csse_covid_19_data/csse_covid_19_time_series'
url += '/time_series_19-covid-Confirmed.csv'
print('Downloading Data from %s' % url)


# In[3]:


confirmed_raw = pd.read_csv(url)


# In[4]:


confirmed_raw.head()


# ### Preprocessing

# In[5]:


confirmed = confirmed_raw[confirmed_raw['Country/Region']=='Germany'].T
confirmed = confirmed[4:].astype('int')
confirmed.columns = ['confirmed']


# In[6]:


confirmed.index = pd.to_datetime(confirmed.index)
confirmed = confirmed.asfreq('D')


# Filter der Daten: Wir nehmen für die Modellbildung erst den Tag als Beginn, an dem erstmals mehr als 100 Erkrankte gemeldet waren.

# In[7]:


confirmed = confirmed[confirmed.confirmed>=100]


# ## Modellvarianten
# 
# Man kann immer ein aktuelles Modell rechnen, oder schauen wie sich die Zahlen verändern, basierend auf einem Modell von einem festen Datum aus.

# In[8]:


#today = confirmed.index[-1] # immer aktuell rechnen
today = datetime.date(2020, 3, 21) # 21.03.2020 als Bezugsdatum nehme


# ## Feature

# In[9]:


confirmed['days'] = (confirmed.index - confirmed.index.min()).days


# In[10]:


confirmed.head()


# ## Ausgangssperren
# 
# Am Wochenende 20.03./21.03.2020 haben einige Gemeinden und Städte Ausgangssperren verhängt (z.B. [Dresden](https://www.dresden.de/media/pdf/presseamt/Allgemeinverfuegung.pdf), Mitterteich, ganz Bayern usw). Daher werden wir uns das Datum mal merken.

# In[11]:


ausgangssperren_timestamp = datetime.date(2020, 3, 21)


# In[12]:


ausgangssperren_timestamp_epoch = time.mktime(ausgangssperren_timestamp.timetuple())*1000


# ## Prediction Model
# 
# Ein exponentielles Wachstum (freie unkontrollierte Ausbreitung) verläuft nach:
# 
# $y = A e^{Bx}$
# 
# Wenn wir das logarithmieren mit dem Log zur Basis $e$, haben wir ein lineares Modell.
# 
# $\log_e(y) = B x + \log_e (A)$

# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


# In[14]:


X = confirmed[:today]['days'].values.reshape(-1, 1)
y = confirmed[:today]['confirmed'].values
logy = np.log(y)


# ### Train

# In[15]:


clf = LinearRegression()
clf.fit(X, logy)


# In[16]:


logy_pred = clf.predict(X)


# Die mit dem linearen Modell vorhergesagten Werte sind im logarithmischen, müssen mit der $e^y$ noch zurück konvertiert werden.

# In[17]:


confirmed.loc[:today,'predicted'] = np.exp(logy_pred).astype('int')


# In[18]:


confirmed.tail()


# ### Modelparameter

# In[19]:


B = clf.coef_[0]
A = np.exp(clf.intercept_)
print('Modellparameter sind A=%.1f, B=%.3f' % (A, B))


# ### Model Evaluation
# 
# R² score: the coefficient of determination

# In[20]:


r2_score(confirmed['confirmed'].values, confirmed['predicted'].values)


# ## Save the model for later use

# In[21]:


import pickle

pklfilename = '%s-Germany-Covid19-Prediction-Model.pkl' % today.strftime('%Y-%m-%d')
with open(pklfilename, 'wb') as f:
    pickle.dump(clf, f)
    
print('Saved the Model to %s' % pklfilename)


# ## Future

# In[22]:


fd = 12 # days into the future


# In[23]:


# Create DataFrame in the Future
dates = pd.date_range(confirmed.index[-1], periods=fd, closed='right')
days_in_future = confirmed.days[-1] + np.arange(1, fd)

future = pd.DataFrame(data=days_in_future, index=dates, columns=['days'])


# In[24]:


ger_future = confirmed.append(future, sort=True)


# ### Predict the Future

# In[25]:


X_future = ger_future['days'].values.reshape(-1, 1)


# In[26]:


logy_pred = clf.predict(X_future)
ger_future['predicted'] = np.exp(logy_pred).astype('int')


# In[27]:


ger_future


# ## Future Plot

# In[28]:


title = 'Bestätigte Fälle und Vorhersage für Deutschland (Basierend auf CSSE COVID-19 Dataset)'


# In[29]:


ax = ger_future['confirmed'].plot(label='Bestätigte COVID-19 Fälle', marker='o')
ax = ger_future['predicted'].plot(label='exponentielles Wachstum\n(Modell vom %s)' % today.strftime('%d.%m.%Y'),
                                  alpha=0.6, ax=ax)

ax.vlines(ausgangssperren_timestamp,
          ymin=ger_future.predicted.min(),
          ymax=ger_future.predicted.max(),
          linestyle='--', alpha=0.2, label='Beginn Ausgangssperren')

ax.legend()
ax.set_ylabel('Log(Anzahl)')
ax.set_yscale('log')
ax.set_title(title, fontsize=8)
ax.annotate('unter CC-BY 2.0 Lizenz Paul Balzer', xy=(.5, 0.02), xycoords='figure fraction', ha='center', fontsize=6, color='gray')

plt.tight_layout()
plt.savefig('./%s-Germany-Covid19-Prediction.png' % today.strftime('%Y-%m-%d'), dpi=150)
print('Saved the Figure')


# ## Export as Excel

# In[30]:


xlsfile = './%s-Germany-Covid19-Prediction.xlsx' % today.strftime('%Y-%m-%d')
ger_future.to_excel(xlsfile)
print('Saved the Excel to %s' % xlsfile)


# # Interactive Website
# 
# We are using Bokeh to export an interactive website

# In[31]:


from bokeh.plotting import figure
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models import Div, HoverTool, BoxAnnotation, Span
from bokeh.layouts import column
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.palettes import inferno


# In[32]:


colors = inferno(6) # What else for this scenario ;)


# In[38]:


p = figure(tools="hover,save,pan,box_zoom,reset,wheel_zoom",
           x_axis_type="datetime", y_axis_type="log",
           title=title.replace(')', ' der John Hopkins University)'))

# Vertical line for Ausgangssperren
vline = Span(location=ausgangssperren_timestamp_epoch,
             dimension='height', line_color='gray',
             line_dash='dashed', line_width=3, name='Beginn Ausgangssperren')
p.add_layout(vline)

# Vorhersagemodell als Linie
p.line(ger_future.index, ger_future.predicted, line_width=4, line_color=colors[3],
       legend='exponentielles Wachstum\n(Modell vom %s)' % today.strftime('%d.%m.%Y'))

# Tatsächliche Fälle als Punkte
p.circle(confirmed.index, confirmed.confirmed, line_color=colors[4],
         fill_color=colors[5], size=14, legend='Bestätigte COVID-19 Fälle')

# X-Achse ordentlich formatieren
p.xaxis.formatter=DatetimeTickFormatter(
    years="%d.%m.%Y",
    months="%d.%m.%Y",
    days="%A %d.%m.%Y",
)

p.yaxis.axis_label = 'Anzahl (Logarithmisch)'

# Daten-Zeitraum
gray_box = BoxAnnotation(left=confirmed.index[0],
                          right=confirmed.index[-1],
                          fill_color='gray', fill_alpha=0.1)
p.add_layout(gray_box)

# Tooltips
p.select_one(HoverTool).tooltips = [
    ('Datum', '@x{%d.%m.%Y}'),
    ('Fälle', '@y{0.0a}'),
]
p.select_one(HoverTool).formatters = {'x':'datetime'}
p.select_one(HoverTool).mode = 'vline'

p.toolbar.autohide = True

# Legende
p.legend.location = "top_left"

# Anmerkung
div = Div(text="""Quellcode: <a href="https://github.com/balzer82/covid-germany-predictor">Covid Germany Predictor</a>
            unter CC-BY2.0 Lizenz von Paul Balzer on Github.
            Disclaimer: Ich bin kein Epidemiologe oder Virologe, das ist keine offizielle Vorausberechnung! 
            <a href="https://cbcity.de/impressum">Impressum</a>""")

# Save
html = file_html(column(p, div, sizing_mode="stretch_both"), CDN, 'COVID-19 Prediction Germany')


# ## Style the Website

# In[39]:


head = '''
  <body>
        <div><img src="https://wirvsvirushackathon.org/wp-content/uploads/2020/03/12-scaled.jpg" style="height: 30px; padding: 10px;"></div>
'''


# In[40]:


gtc = '''
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-7230698-7"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'UA-7230698-7');
        </script>
    </body>
'''


# In[41]:


websitehtml = html.replace('<body>', head)
websitehtml = websitehtml.replace('</body>', gtc)


# In[42]:


with open('./html/index.html', 'w') as htmlfile:
    htmlfile.write(websitehtml.strip())

print('Saved the /html/index.html')


# CC-BY 2.0 Paul Balzer
