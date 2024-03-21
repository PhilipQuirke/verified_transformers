import re
from abc import ABC, abstractmethod

from .quanta_constants import QCondition, QType, MIN_ATTENTION_PERC 

from .useful_node import position_name, position_name_to_int, UsefulNodeList 


def extract_trailing_int(input_string):
    # Regular expression to find trailing integer
    match = re.search(r'\d+$', input_string)
    if match:
        return int(match.group())
    else:
        return None
        

class FilterNode(ABC):
    @abstractmethod
    def evaluate(self, test_node):
        pass

    @abstractmethod
    def describe(self):
        pass

                                       
class FilterAnd(FilterNode):
    def __init__(self, *args):
        self.children = args

    def evaluate(self, test_node):
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

    def evaluate(self, test_node):
        return any(child.evaluate(test_node) for child in self.children)

    def describe(self):
        answer = " or( "
        for child in self.children:
            answer += child.describe() + ", "
        answer = answer[:-2]  # Remove the last two characters
        answer += ")"
        return answer    


class FilterHead(FilterNode):

    def evaluate(self, test_node):
        return test_node.is_head

    def describe(self):
        return "IsHead" 

                                       
class FilterNeuron(FilterNode):

    def evaluate(self, test_node):
        return not test_node.is_head

    def describe(self):
        return "IsNeuron" 

                                       
class FilterPosition(FilterNode):
    def __init__(self, the_position_name, filter_strength = QCondition.MUST):
        self.filter_strength = filter_strength
        self.position = position_name_to_int(the_position_name)

    def evaluate(self, test_node):
        if self.filter_strength in [QCondition.MUST, QCondition.CONTAINS]:
            return (test_node.position == self.position)
        if self.filter_strength == QCondition.NOT:
            return (not test_node.position == self.position)
        if self.filter_strength == QCondition.MAY:
            return True 
        if self.filter_strength == QCondition.MUST_BY:
            return (test_node.position <= self.position)      
        return False

    def describe(self):
        return self.filter_strength.value + " " + position_name(self.position)

                                       
class FilterContains(FilterNode):
    def __init__(self, quanta_type, minor_tag, filter_strength = QCondition.MUST):
        self.quanta_type = quanta_type
        self.minor_tag = minor_tag
        self.filter_strength = filter_strength

    def evaluate(self, test_node):
        if self.filter_strength in [QCondition.MUST, QCondition.CONTAINS]:
            return test_node.contains_tag(self.quanta_type, self.minor_tag)   
        if self.filter_strength == QCondition.NOT:
            return not test_node.contains_tag(self.quanta_type, self.minor_tag)
        if self.filter_strength == QCondition.MAY:
            return True  
        return False

    def describe(self):
        return self.filter_strength.value + " " + str(self.quanta_type) + " " + self.minor_tag


class FilterAttention(FilterContains):
    def __init__(self, minor_tag, filter_strength = QCondition.MUST, filter_min_perc = MIN_ATTENTION_PERC):
        super().__init__(QType.ATTENTION, minor_tag, filter_strength)
        self.filter_min_perc = filter_min_perc

    def evaluate(self, test_node):
        if self.filter_strength in [QCondition.MUST, QCondition.CONTAINS]:
            for tag in test_node.tags:
                # We use contains(minor) as the ATTENTION_MAJOR_TAG minor tag is "P14=25" (i.e 25 percent)
                if tag.startswith(str(QType.ATTENTION)) and (self.minor_tag in tag) and (extract_trailing_int(tag)>=self.filter_min_perc):
                    return True
            
        return super().evaluate(test_node)
        

class FilterImpact(FilterContains):
    def __init__(self, minor_tag, filter_strength = QCondition.MUST):
        super().__init__(QType.IMPACT, minor_tag, filter_strength)


class FilterPCA(FilterContains):
    def __init__(self, minor_tag, filter_strength = QCondition.MUST):
        super().__init__(QType.PCA, minor_tag, filter_strength)


class FilterAlgo(FilterContains):
    def __init__(self, minor_tag, filter_strength = QCondition.MUST):
        super().__init__(QType.ALGO, minor_tag, filter_strength)



# Filters the list of nodes using the specified filter criteria and returns a (likely smaller) list of nodes.
def filter_nodes( the_nodes : UsefulNodeList, the_filters: FilterNode):
    answer = UsefulNodeList()
    
    for test_node in the_nodes.nodes:
        if the_filters.evaluate(test_node):
            answer.nodes.append(test_node)

    answer.sort_nodes()
    
    return answer
