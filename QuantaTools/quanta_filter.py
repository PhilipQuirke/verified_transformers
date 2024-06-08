import re
from abc import ABC, abstractmethod

from .quanta_constants import QCondition, QType, MIN_ATTN_PERC 

from .useful_node import position_name, position_name_to_int, UsefulNode, UsefulNodeList 


def extract_trailing_int(input_string):
    # Regular expression to find trailing integer
    match = re.search(r'\d+$', input_string)
    if match:
        return int(match.group())
    else:
        return None
        

class FilterNode(ABC):
    @abstractmethod
    def evaluate(self, test_node:UsefulNode):
        pass

    @abstractmethod
    def describe(self):
        pass

                                       
class FilterAnd(FilterNode):
    def __init__(self, *args):
        self.children = args

    def evaluate(self, test_node:UsefulNode):
        return all(child.evaluate(test_node) for child in self.children)

    def describe(self):
        answer = " and( "
        for child in self.children:
            answer += child.describe() + ", "
        answer = answer[:-2]  # Remove the last two characters            
        answer += ")"
        return answer


class FilterOr(FilterNode):
    def __init__(self, *args):
        self.children = args

    def evaluate(self, test_node:UsefulNode):
        return any(child.evaluate(test_node) for child in self.children)

    def describe(self):
        answer = " or( "
        for child in self.children:
            answer += child.describe() + ", "
        answer = answer[:-2]  # Remove the last two characters
        answer += ")"
        return answer    


class FilterName(FilterNode):
    def __init__(self, name:str):
        self.name = name
        
    def evaluate(self, test_node:UsefulNode):
        return test_node.name() == self.name

    def describe(self):
        return "Name=" + self.name
    

class FilterTrue(FilterNode):
    def __init__(self):
        pass
        
    def evaluate(self, test_node:UsefulNode):
        return True

    def describe(self):
        return "True"
    

class FilterHead(FilterNode):

    def evaluate(self, test_node:UsefulNode):
        return test_node.is_head

    def describe(self):
        return "IsHead" 

                                       
class FilterNeuron(FilterNode):

    def evaluate(self, test_node:UsefulNode):
        return not test_node.is_head

    def describe(self):
        return "IsNeuron" 

                                       
class FilterPosition(FilterNode):
    def __init__(self, the_position_name, filter_strength = QCondition.MUST):
        self.filter_strength = filter_strength
        self.position = position_name_to_int(the_position_name)

    def evaluate(self, test_node:UsefulNode):
        if self.filter_strength in [QCondition.MUST, QCondition.CONTAINS]:
            return (test_node.position == self.position)
        if self.filter_strength == QCondition.NOT:
            return (not test_node.position == self.position)
        if self.filter_strength == QCondition.MAY:
            return True 
        if self.filter_strength == QCondition.MAX:
            return (test_node.position <= self.position)      
        if self.filter_strength == QCondition.MIN:
            return (test_node.position >= self.position)      
        return False

    def describe(self):
        return self.filter_strength.value + " " + position_name(self.position)


class FilterLayer(FilterNode):
    def __init__(self, the_layer : int):
        self.layer = the_layer

    def evaluate(self, test_node:UsefulNode):
        return (test_node.layer == self.layer)

    def describe(self):
        return "Layer=" + str(self.layer)

                                       
class FilterContains(FilterNode):
    def __init__(self, quanta_type : QType, minor_tag : str, filter_strength = QCondition.MUST):
        self.quanta_type = quanta_type
        self.minor_tag = minor_tag
        self.filter_strength = filter_strength

    def evaluate(self, test_node:UsefulNode):
        if self.filter_strength in [QCondition.MUST, QCondition.CONTAINS]:
            return test_node.contains_tag(self.quanta_type.value, self.minor_tag)   
        if self.filter_strength == QCondition.NOT:
            return not test_node.contains_tag(self.quanta_type.value, self.minor_tag)
        if self.filter_strength == QCondition.MAY:
            return True  
        return False

    def describe(self):
        return self.filter_strength.value + " " + self.quanta_type.value + " " + self.minor_tag


class FilterAttention(FilterContains):
    def __init__(self, minor_tag, filter_strength = QCondition.MUST, filter_min_perc = MIN_ATTN_PERC):
        super().__init__(QType.ATTN, minor_tag, filter_strength)
        self.filter_min_perc = filter_min_perc

    def evaluate(self, test_node:UsefulNode):
        if self.filter_strength in [QCondition.MUST, QCondition.CONTAINS]:
            for tag in test_node.tags:
                # We use contains(minor) as the ATTENTION_MAJOR_TAG minor tag is "P14=25" (i.e 25 percent)
                if tag.startswith(str(QType.ATTN)) and (self.minor_tag in tag) and (extract_trailing_int(tag)>=self.filter_min_perc):
                    return True
            
        return super().evaluate(test_node)
        

class FilterImpact(FilterContains):
    def __init__(self, minor_tag : str = "", filter_strength = QCondition.MUST):
        super().__init__(QType.IMPACT, minor_tag, filter_strength)


class FilterAlgo(FilterContains):
    def __init__(self, minor_tag : str = "", filter_strength = QCondition.MUST):
        super().__init__(QType.ALGO, minor_tag, filter_strength)


# Filters the list of nodes using the specified filter criteria and returns a (likely smaller) list of nodes.
def filter_nodes(the_nodes : UsefulNodeList, the_filters: FilterNode):
    answer = UsefulNodeList()
    
    for test_node in the_nodes.nodes:
        if the_filters.evaluate(test_node):
            answer.nodes.append(test_node)

    answer.sort_nodes()
    
    return answer


# Show the fraction of useful nodes that have an assigned algorithmic purpose
def print_algo_purpose_results(cfg):
    the_nodes = cfg.useful_nodes
    num_heads = the_nodes.num_heads
    num_neurons = the_nodes.num_neurons

    algo_nodes = filter_nodes( the_nodes, FilterAlgo() )
    num_heads_with_purpose = algo_nodes.num_heads
    num_neurons_with_purpose = algo_nodes.num_neurons

    if num_heads>0:
        print(f"{num_heads_with_purpose} of {num_heads} useful attention heads ({num_heads_with_purpose / num_heads * 100:.2f}%) have an algorithmic purpose assigned." )
    if num_neurons>0:
        print(f"{num_neurons_with_purpose} of {num_neurons} useful MLP neurons ({num_neurons_with_purpose / num_neurons * 100:.2f}%) have an algorithmic purpose assigned." )


