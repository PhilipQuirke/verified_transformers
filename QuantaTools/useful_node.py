import json
import re
from typing import List


# Convert int 14 to "P14"
def position_name(position):
    return "P" + str(position)


# Convert "P14" to int 14
def position_name_to_int(s):
    return int(s.lstrip('P'))


# Return "L1H2" or "L1M0"
def row_location_name(layer, is_head, num):
    return "L" + str(layer) + ("H" if is_head else "M") + str(num) 


# Return "P19L1H2" or "P19L1M0"
def location_name(position, layer, is_head, num):
    return "P" + str(position) + row_location_name(layer, is_head, num)


# Convert 3 to "A3"
def answer_name(n):
    return "A" + str(n)


# The unique location of a node (attention head or MLP neuron) in a model 
class NodeLocation():


    def __init__(self, position : int, layer : int, is_head : bool, num : int):
        # Token position. Zero-based
        self.position = position 
        
        # Layer. Zero-based
        self.layer = layer 
        
        # Is this an attention had or MLP neuron location
        self.is_head = is_head 
        
        # Either attention head or MLP neuron number. Zero-based
        self.num = num 
    

    # Node name e.g. "P14L2H3" or "P14L2M0"
    def name(self):
        return location_name(self.position,self.layer,self.is_head,self.num)


    # Node row name e.g. "L2H3" or "L2M0"
    def row_name(self):
        return row_location_name(self.layer,self.is_head,self.num)


def str_to_node_location( node_location_as_string ):
    pattern = r"P(\d{1,5})L(\d{1,5})H(\d{1,5})"
    match = re.search(pattern, node_location_as_string)
    if match:
        position, layer, num = match.groups()
        return NodeLocation( int(position), int(layer), True, int(num))

    pattern = r"P(\d{1,5})L(\d{1,5})M(\d{1,5})"
    match = re.search(pattern, node_location_as_string)
    if match:
        position, layer, num = match.groups()
        return NodeLocation( int(position), int(layer), False, int(num))
 
    return None


# A UsefulNode contains a NodeLocation and a list of tags representing its behaviour and purpose
class UsefulNode(NodeLocation):


    def __init__(self, position : int, layer : int, is_head : bool, num : int, tags : List[str]):
        super().__init__(position, layer, is_head, num)  

        # Tags related to the node of form "MajorVersion:MinorVersion"  containing behaviour and purpose data
        self.tags = tags

  
    # Remove some/all tags from this 
    def reset_tags(self, major_tag):
        if str(major_tag) == "":
            self.tags = []
        else:
            self.tags = [s for s in self.tags if not s.startswith(str(major_tag))]


    # Add a tag to this  (if not already present)
    def add_tag(self, major_tag, minor_tag):
        assert str(major_tag) != ""
    
        tag = str(major_tag) + ":" + minor_tag
        if tag != "" and (not (tag in self.tags)):
            self.tags += [tag]


    # Return tags with the matching major and minor versions
    def filter_tags(self, major_tag, minor_tag = ""):
        assert str(major_tag) != ""

        filtered_strings = [s for s in self.tags if s.startswith(str(major_tag))]

        minor_tags = [s.split(":")[1] for s in filtered_strings]

        if minor_tag != "":
            minor_tags = [s for s in minor_tags if s.startswith(minor_tag)]

        return minor_tags


    # Return minimum tag with the matching major and minor versions
    def min_tag_suffix(self, major_tag, minor_tag = ""):
        assert str(major_tag) != ""

        minor_tags = self.filter_tags(major_tag)

        if minor_tag != "":
            minor_tags = [s for s in minor_tags if s.startswith(minor_tag)]

        return min(minor_tags) if minor_tags else ""


    # Return the only tag with the matching major_tag
    def only_tag(self, major_tag):
        assert str(major_tag) != ""

        filtered_strings = [s for s in self.tags if s.startswith(str(major_tag))]

        num_strings = len(filtered_strings)
        if num_strings > 1:
            print("only_tag logic failure", str(major_tag), num_strings, filtered_strings)
            assert False

        return filtered_strings[0].split(":")[1] if num_strings == 1 else ""


    # Return whether this  contains a tag with the matching major_tag or major+minor_tag
    def contains_tag(self, major_tag, minor_tag):
        assert str(major_tag) != ""

        for tag in self.tags:
            # We use contains(minor) as the ATTENTION_MAJOR_TAG minor tag is "P14=25" (i.e 25 percent)
            if tag.startswith(str(major_tag)) and minor_tag in tag:
                return True

        return False


    def to_dict(self):
        return {
          "position": self.position,
          "layer": self.layer,
          "is_head": self.is_head,
          "num": self.num,
          "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data['position'], data['layer'], data['is_head'], data['num'], data['tags'])


class UsefulNodeList():

  
    def __init__(self):
        # list of useful (attention head and MLP neuron) nodes
        self.nodes = []

  
    def get_node_names(self):
        answer = ""
        for node in self.nodes:
            answer += ( "" if answer == "" else ", " ) + node.name()
        return answer

  
    def num_heads(self):
        answer = 0
        for node in self.nodes:
            answer += 1 if node.is_head else 0
        return answer


    def num_neurons(self):
        return len(self.nodes) - self.num_heads()
    

    def print_node_tags(self, major_tag = "", minor_tag = "", show_empty_tags = True):
        for node in self.nodes:
            tags = node.tags if str(major_tag) == "" else node.filter_tags(major_tag, minor_tag)
            if show_empty_tags or len(tags) > 0 :
                print( node.name(), tags )       


    def reset_node_tags( self, major_tag = "" ):
        for node in self.nodes:
            node.reset_tags(major_tag)


    # Get the node at the specified location. May return None.    
    def get_node( self, nodelocation ):
        for node in self.nodes:
            if node.position == nodelocation.position and node.is_head == nodelocation.is_head and node.layer == nodelocation.layer and node.num == nodelocation.num:
                return node

        return None


    # Add the tag to the node location (creating a node if necessary)  
    def add_node_tag( self, nodelocation, major_tag, minor_tag ):

        the_node = self.get_node( nodelocation )
        if the_node == None:

            the_node = UsefulNode(nodelocation.position, nodelocation.layer, nodelocation.is_head, nodelocation.num, [])

            self.nodes += [the_node]

        the_node.add_tag(major_tag, minor_tag)


    # Sort the nodes into position, layer, is_head, num order
    def sort_nodes(self):
        self.nodes = sorted(self.nodes, key=lambda obj: (obj.position, obj.layer, obj.is_head, obj.num))


    # Save the nodes and tags to a json file
    def save_nodes(self, filename):
        dict_list = [node.to_dict() for node in self.nodes]
        with open(filename, 'w') as file:
            json.dump(dict_list, file, default=lambda o: o.__dict__)

