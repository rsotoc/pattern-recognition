# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 20:49:40 2017

@author: rsotoc
"""

#Packages
import pandas as pd
import numpy as np
#from numpy.random import random
#from math import ceil
#from pandas.compat import StringIO
#from pandas.io.common import urlopen
#from IPython.display import display, display_pretty, Javascript, HTML
#from matplotlib.path import Path
#from matplotlib.spines import Spine
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
#from plotly.tools import FigureFactory as FF
#from plotly import tools
#import plotly
#import plotly.plotly as py
#import seaborn as sns


# Aux variables
colorIdentity_map = {'B': 'Black', 'G': 'Green', 'R': 'Red', 'U': 'Blue', 'W': 'White'}
keeps = ['name', 'colorIdentity', 'colors', 'type', 'types', 'cmc', 'power', 'toughness', 'legalities']

#Data reading
raw = pd.read_json("AllSets-x.json")

# Data fusion
mtg = []
for col in raw.columns.values:
    release = pd.DataFrame(raw[col]['cards'])
    release = release.loc[:, keeps]
    release['releaseName'] = raw[col]['name']
    release['releaseDate'] = raw[col]['releaseDate']
    mtg.append(release)
mtg = pd.concat(mtg)

del release, raw

# Combine colorIdentity and colors
mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colors'] = mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colorIdentity'].apply(lambda x: [colorIdentity_map[i] for i in x])
mtg['colorsCount'] = 0
mtg.loc[mtg.colors.notnull(), 'colorsCount'] = mtg.colors[mtg.colors.notnull()].apply(len)
mtg.loc[mtg.colors.isnull(), 'colors'] = ['Colorless']
mtg['colorsStr'] = mtg.colors.apply(lambda x: ''.join(x))

# Include colorless and multi-color.
mtg['manaColors'] = mtg['colorsStr']
mtg.loc[mtg.colorsCount>1, 'manaColors'] = 'Multi'

# Set Date type for the releaseDate
format = '%Y-%m-%d'
mtg['releaseDate'] = pd.to_datetime(mtg['releaseDate'], format=format)

# Remove promo cards that aren't used in normal play
mtg_nulls = mtg.loc[mtg.legalities.isnull()]
mtg = mtg.loc[~mtg.legalities.isnull()]

# Remove cards that are banned in any game type
mtg = mtg.loc[mtg.legalities.apply(lambda x: sum(['Banned' in i.values() for i in x])) == 0]
mtg = pd.concat([mtg, mtg_nulls])
mtg.drop('legalities', axis=1, inplace=True)
del mtg_nulls

# Remove tokens without types
mtg = mtg.loc[~mtg.types.apply(lambda x: isinstance(x, float))]

# Transform types to str
mtg.loc[mtg.types.notnull(), 'typesCount'] = mtg.types[mtg.types.notnull()].apply(len)
mtg['typesStr'] = mtg.types.apply(lambda x: ''.join(x))

# Power and toughness that depends on board state or mana cannot be resolved
mtg[['power', 'toughness']] = mtg[['power', 'toughness']].apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Include colorless and multi-color.
mtg['manaColors'] = mtg['colorsStr']
mtg.loc[mtg.colorsCount>1, 'manaColors'] = 'Multi'

# Remove 'Gleemax' and other cards with more than 90 cmc 
mtg = mtg[(mtg.cmc < 90) | (mtg.cmc.isnull())]

# Remove 'Big Furry Monster' and other cards with more than 90 of power and toughness
mtg = mtg[(mtg.power < 90) | (mtg.typesStr != 'Creature')]
mtg = mtg[(mtg.toughness < 90) | (mtg.typesStr != 'Creature')]

# Remove 'Spinal Parasite' and other cards whose power and toughness depends on the number of lands used to cast it
mtg = mtg[(mtg.power > 0) | (mtg.typesStr != 'Creature')]
mtg = mtg[(mtg.toughness > 0) | (mtg.typesStr != 'Creature')]
          
# Remove the duplicated cards
duplicated = mtg[mtg.duplicated(['name'])]
mtg = mtg.drop_duplicates(['name'], keep='first')

# Recode the card type 'Eaturecray' (Atinlay Igpay), which means 'Creature' on Pig-latin
mtg['typesStr'] = mtg['typesStr'].replace('Eaturecray', 'Creature')

cards_recoded_absolutes=(len(mtg[mtg.typesStr=='Vanguard']) + len(mtg[mtg.typesStr=='Scheme']) + len(mtg[mtg.typesStr=='Plane']) + len(mtg[mtg.typesStr=='Phenomenon']) + len(mtg[mtg.typesStr=='Conspiracy']))
cards_recoded_relatives=str(round((((float(len(mtg[mtg.typesStr=='Vanguard']) + len(mtg[mtg.typesStr=='Scheme']) + len(mtg[mtg.typesStr=='Plane']) + len(mtg[mtg.typesStr=='Phenomenon']) + len(mtg[mtg.typesStr=='Conspiracy']))) / float(len(mtg))) * 100), 2))+'%'

# Recode some special card types to 'Other types'
mtg = mtg.replace(['Vanguard', 'Scheme', 'Plane', 'Phenomenon', 'Conspiracy'], 'Other types')

# Transform the multi-choice variable 'types' to a 7-item dichotomized variable
mtg.loc[mtg.types.notnull(), 'typesCount'] = mtg.types[mtg.types.notnull()].apply(len)
mono_types = mtg[mtg.typesCount==1]
mono_types = np.sort(mono_types.typesStr.unique()).tolist()
for types in mono_types:
    mtg[types] = mtg.types.apply(lambda x: types in x)
    
#Transform the multi-choice variable 'colors' to a 5-item dichotomized variable
mono_colors = np.sort(mtg.colorsStr[mtg.colorsCount==1].unique()).tolist()
for color in mono_colors:
    mtg[color] = mtg.colors.apply(lambda x: color in x)
    
    
# Get the data
cards_over_time = pd.pivot_table(mtg, values='name',index='releaseDate', aggfunc=len)
cards_over_time.fillna(0, inplace=True)
cards_over_time = cards_over_time.sort_index()

#Create a trace
trace = go.Scatter(x=cards_over_time.index,
                   y=cards_over_time.values)

# Create the range slider
data = [trace]
layout = dict(
    title="Number of new (unique) cards over time",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

# Plot the data
fig = dict(data=data, layout=layout)
#plotly.offline.iplot(fig)