<H3>Name: FARHANA H</H3>
<H3>Register No: 212223230057</H3>
<H3>Experiment: 1</H3>
<H3>Date:12.05.2026</H3>
<H1 ALIGN=CENTER> Implementation of Bayesian Networks</H1>

## Aim:
To create a Bayesian Network for the given dataset in Python

## Algorithm:
<b>Step 1:</b> Import necessary libraries: pandas, networkx, matplotlib.pyplot, Bbn, Edge, EdgeType, BbnNode, Variable, EvidenceBuilder, InferenceController

<b>Step 2:</b> Set pandas options to display more columns

<b>Step 3:</b> Read in weather data from a CSV file using pandas

<b>Step 4:</b> Remove records where the target variable RainTomorrow has missing values

<b>Step 5:</b> Fill in missing values in other columns with the column mean

<b>Step 6:</b> Create bands for variables that will be used in the model (Humidity9amCat, Humidity3pmCat, and WindGustSpeedCat)

<b>Step 7:</b> Define a function to calculate probability distributions, which go into the Bayesian Belief Network (BBN)

<b>Step 8:</b> Create BbnNode objects for Humidity9amCat, Humidity3pmCat, WindGustSpeedCat, and RainTomorrow, using the probs() function to calculate their probabilities

<b>Step 9:</b> Create a Bbn object and add the BbnNode objects to it, along with edges between the nodes

<b>Step 10:</b> Convert the BBN to a join tree using the InferenceController

<b>Step 11:</b> Set node positions for the graph

<b>Step 12:</b> Set options for the graph appearance

<b>Step 13:</b> Generate the graph using networks

<b>Step 14:</b> Update margins and display the graph using matplotlib.pyplot

## Program:
```python
# Import required libraries
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pybbn.graph.dag import Bbn
from pybbn.graph.dag import Edge,EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
pd.options.display.max_columns=50

# Read the dataset
df=pd.read_csv('weatherAUS.csv',encoding='utf-8')

# Remove row which contain output(RainTomorrow) NA
df=df[pd.isnull(df['RainTomorrow'])==False]

# Filling missing values
df=df.fillna(df.mean())

# Creating bands for variable
df['WindGustSpeedCat']=df['WindGustSpeed'].apply(lambda x: '0.<=40'   if x<=40 else '1.40-50' if 40<x<=50 else '2.>50')
df['Humidity9amCat']=df['Humidity9am'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
df['Humidity3pmCat']=df['Humidity3pm'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
print(df)

# Function to calculate probabilities
def probs(data, child, parent1=None, parent2=None):
    if parent1==None:
        # Calculate probabilities
        prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent1!=None:
            # Check if child node has 1 parent or 2 parents
            if parent2==None:
                # Caclucate probabilities
                prob=pd.crosstab(data[parent1],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else:
                # Caclucate probabilities
                prob=pd.crosstab([data[parent1],data[parent2]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else: print("Error in Probability Frequency Calculations")
    return prob

# Creating BBN network
H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))

bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))

join_tree = InferenceController.apply(bbn)
pos={0: (-1,0), 1: (-1, 0.5), 2: (1, 0), 3:(0,-0.5)}

options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "yellow",
    "edgecolors": "green",
    "edge_color": "black",
    "linewidths": 5,
    "width": 5,}


n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()
```

## Output:
<img width="388" height="563" alt="image" src="https://github.com/user-attachments/assets/ae4d47e6-1c79-4c5c-86a4-b07cc4854ed3" />
<img width="481" height="471" alt="image" src="https://github.com/user-attachments/assets/81170ece-eaa3-4117-93d9-36eaba4fc10e" />

## Result:
Thus, a Bayesian Network is generated using Python

