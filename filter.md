# Node Filters
You can filter the useful nodes based on their tags, using the Filter classes. 

## Filter Types
The generic Filters are:
- FilterAnd: node must satisfy all the child criteria to be selected 
- FilterOr: node must satisfy at least one child criteria to be selected
- FilterHead: node must be an attention head
- FilterNeuron: node must be an MLP neuron
- FilterPosition: node must be located the the specified location
- FilterContains: node tags must contain the specified text
- FilterAttention: node must attend to the specified token and (optionally) with at least the specified percentage strength 
- FilterImpact: node must impact the specified answer token(s)
- FilterPCA: node must have a PCA - meaning the PAC results are interpretable
- FilterAlgo: node must have the specified algorithm tag

The library can be extended with topic-additional specific filter classes

## Filter Use 
Filters are often used to find candidate nodes that could implement a specific algorithmic task. 
Confirming that a node implements a specific algorithmic task is not covered here.
If we find a candidate node **does** implement the task, a new tag is added to the node to document that fact.   

As an example, suppose we have an addition model, and want to find all "candidate" nodes that:
- Are at position P14
- Are an attention head
- Attend to (at least) D2 and D'2
- Impact (at least) answer token A2

In a Colab, we can find the candidate nodes with this filter:

````import QuantaTools as qt

my_filters = qt.FilterAnd(
    qt.FilterHead(),
    qt.FilterPosition(qt.position_name(14)),
    qt.FilterAttention(cfg.dn_to_position_name(2)), # Attends to Dn
    qt.FilterAttention(cfg.ddn_to_position_name(2)), # Attends to D'n
    qt.FilterImpact(qt.answer_name(2))) # Impacts An

test_nodes = qt.filter_nodes(cfg.useful_nodes, my_filters)
````
