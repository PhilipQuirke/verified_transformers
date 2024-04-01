# Useful nodes and tags

## Calculation Location Syntax: PnnLnHn and PnnLnMn 
A transformer model is partially categorised by the number of layers. 
Each layer has a number of attention heads (aka heads) and a number of MLP neurons (aka neurons) where it can do calculations.
A model is given a question and asked to predict the answer. The question contains a number of question (aka input) tokens. 
The answer contains a number of answer (aka output) tokens.
Calculations occur at each (input and output) token position, at each layer, and at each head and each neuron.
We term these locations PnnLnHn for attention heads and PnnLnMn for MLP neurons. All numbers are zero-based.

## Useful Calculation Locations 
When a transformer model performs a task it was trained to do, the model usually only relies on **some** token positions e.g. P0, P8, P12..P18.
Token positions that are used in calculations are termed **useful** and are stored in **UsefulConfig.useful_locations**

## Useful Calculation Nodes
At these useful locations, only some of the the model usually only relies on **some** of the available nodes e.g. at P8 the model may rely on P8L0H0 and P8L0H1 but not P8L0H2.     
Nodes that are used in calculations are termed **useful** and are stored in **UsefulConfig.useful_nodes**

## Finding Useful Nodes
The library find the useful locations and nodes by asking the model to predict questions with known answers. 
Model nodes are systematically ablated (aka removed) to see whether the model needs the nodes to do calculations.

## Percentage Failures (aka Fail)
The percentage of questions that fail when each useful node is ablated can be displayed.   
Each colored cell represents a useful node. A high percentage (in the cell) means that this node is used in most predictions.
The node's location can be read as the position (e.g. P6) as shown at the bottom on the x axis, followed by the layer (e.g. L0H0) as shown at the left on the y axis.
![FailureRate](./assets/ins1_mix_d6_l3_h4_t40K_s372001FailureFrequencyBehaviorPerNode.svg?raw=true "FailureRate")

The column headings at top of the diagram are the position **meaning** and specific to the particular model being studied. 
This diagram came from the maths model which has question and answer token format
D5 D4 D3 D2 D1 D0 + D'5 D'4 D'3 D'2 D'1 D'0 = A7 A6 A5 A4 A3 A2 A1 A0

## Answer Impact (aka Impact)
The answer tokens that the model gets wrong when each useful node is ablated can be displayed. If a cell contains "A7..4" this means that when this cell is ablated, the model gets answer digits A7, A6, A5 and A4 wrong. That is, this cell is needed to get these answer tokens correct. 
![AnswerImpact](./assets/ins1_mix_d6_l3_h4_t40K_s372001AnswerImpactBehaviorPerNode.svg?raw=true "AnswerImpact")

## Useful Node Tags
The list of useful nodes and the facts gathered about them (such as percentage failure and answer impact) are saved to a JSON file:

```P0L0H3 ['Fail:3', 'Impact:A7', 'MathSub:M0', 'Attn:P0=100']
P6L0H0 ['Fail:3', 'Impact:A43210', 'MathSub:M123', 'Attn:P1=74', 'Attn:P6=22', 'Attn:P4=2', 'Attn:P2=1', 'PCA:A0.PA.Weak', 'PCA:A1.PA.Weak', 'PCA:A4.PA.Weak', 'PCA:A5.PA.Weak']
P9L0H0 ['Fail:4', 'Impact:A765', 'MathAdd:S1234', 'MathSub:M3', 'Attn:P8=50', 'Attn:P1=49', 'PCA:A3.PA.Weak', 'PCA:A5.PA.Weak', 'PCA:A0.PS.Weak', 'PCA:A3.PS.Weak', 'PCA:A5.PS.Weak']
P9L0H1 ['Fail:1', 'Impact:A654', 'MathAdd:S124', 'Attn:P8=52', 'Attn:P1=46', 'PCA:A0.PA.Weak', 'PCA:A1.PA.Weak', 'PCA:A2.PA.Weak', 'PCA:A3.PA.Weak', 'PCA:A0.PS.Weak', 'PCA:A1.PS.Weak', 'PCA:A2.PS.Weak', 'PCA:A3.PS.Weak']
P9L0H3 ['Fail:0', 'Impact:A7', 'MathSub:NG', 'Attn:P6=84', 'Attn:P9=6', 'Attn:P8=3', 'Attn:P7=2', 'PCA:A3.PA.Weak', 'PCA:A4.PA.Weak', 'PCA:A5.PA.Weak']
P9L0MLP ['Fail:11', 'Impact:A765', 'MathAdd:S12345', 'MathSub:M123']
```





