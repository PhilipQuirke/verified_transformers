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

## Behaviour Maps
The library find the useful locations and nodes by asking the model to predict questions with known answers. 
Model nodes are systematically ablated (aka removed) to see whether the model needs the nodes to do calculations.
This search can be mapped (by this library) as follows:

![FailureRate](./Static/ins1_mix_d6_l3_h4_t40K_s372001FailureFrequencyBehaviorPerNode.svg?raw=true "FailureRate")
