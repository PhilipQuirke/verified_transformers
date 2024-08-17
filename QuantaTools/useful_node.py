import json
import re
from typing import List


# Convert int 14 to "P14"
def position_name(position : int):
    return "P" + str(position)


# Convert "P14" to int 14
def position_name_to_int(s):
    return int(s.lstrip('P'))


# Return "L1H2" or "L1M0"
def row_location_name(layer : int, is_head : bool, num : int):
    return "L" + str(layer) + (("H" + str(num)) if is_head else "MLP")   # Paper 2
    # return "L" + str(layer) + ("H" if is_head else "M") + str(num)  # Paper 3


# Return "P19L1H2" or "P19L1M0"
def location_name(position : int, layer : int, is_head : bool, num : int, short_position = True):
    pos_str = str(position) 
    if not short_position and len(pos_str)<2:
        pos_str = "0" + pos_str    

    return "P" + pos_str + row_location_name(layer, is_head, num)


# Convert 3 to "A3"
def answer_name(n : int):
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
    def name(self, short_position = True ):
        return location_name(self.position,self.layer,self.is_head,self.num, short_position)


    @property
    # Node row name e.g. "L2H3" or "L2M0"
    def row_name(self) -> str:
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
    def reset_tags(self, major_tag : str, minor_tag : str = ""):
        if major_tag == "":
            self.tags = []
        elif minor_tag == "":
            self.tags = [s for s in self.tags if not s.startswith(major_tag)]
        else:
            self.tags = [s for s in self.tags if not (s.startswith(major_tag) and (minor_tag in s[len(major_tag):])) ]


    # Add a tag to this (if not already present). Returns number of tags added as 0 or 1
    def add_tag(self, major_tag : str, minor_tag : str):
        assert major_tag != ""
    
        tag = major_tag + ":" + minor_tag
        if tag != "" and (not (tag in self.tags)):
            self.tags += [tag]
            return 1
        
        return 0


    # Return tags with the matching major and minor versions
    def filter_tags(self, major_tag : str, minor_tag : str = ""):
        assert major_tag != ""

        filtered_strings = [s for s in self.tags if s.startswith(major_tag)]

        minor_tags = [s.split(":")[1] for s in filtered_strings]

        if minor_tag != "":
            minor_tags = [s for s in minor_tags if s.startswith(minor_tag)]

        return minor_tags


    # Return minimum tag with the matching major and minor versions
    def min_tag_suffix(self, major_tag : str, minor_tag : str = ""):
        assert major_tag != ""

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
            # We use contains(minor) as the QType.ATTN minor tag is "P14=25" (i.e 25 percent)
            if tag.startswith(str(major_tag)) and minor_tag in tag:
                return True

        return False


    def to_dict(self, major_tag = ""):
        the_tags = self.tags    
        if major_tag != "":
            the_tags = [s for s in self.tags if s.startswith(major_tag)]

        return {
          "position": self.position,
          "layer": self.layer,
          "is_head": self.is_head,
          "num": self.num,
          "tags": the_tags
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data['position'], data['layer'], data['is_head'], data['num'], data['tags'])


class UsefulNodeList():

  
    # Create an empty list of useful nodes
    def __init__(self):
        # list of useful (attention head and MLP neuron) nodes
        self.nodes = []

  
    @property
    # Return the node names as a comma-separated string
    def node_names(self) -> str:
        answer = ""
        for node in self.nodes:
            answer += ( "" if answer == "" else ", " ) + node.name()
        return answer

  
    @property
    # Number of nodes that are attention heads
    def num_heads(self) -> int:
        answer = 0
        for node in self.nodes:
            answer += 1 if node.is_head else 0
        return answer


    @property
    # Number of nodes that are MLP neurons
    def num_neurons(self) -> int:
        return len(self.nodes) - self.num_heads
    

    # Print the node names and tags matching major_tag and minor_tag (if specified).
    def print_node_tags(self, major_tag : str = "", minor_tag : str = "", show_empty_tags : bool = True):
        for node in self.nodes:
            tags = node.tags if str(major_tag) == "" else node.filter_tags(major_tag, minor_tag)
            if show_empty_tags or len(tags) > 0 :
                print( node.name(), tags )       


    # Delete all tags matching major_tag (if specified) from all nodes. Else delete all tags.
    def reset_node_tags( self, major_tag : str = "", minor_tag : str = "" ):
        for node in self.nodes:
            node.reset_tags(major_tag, minor_tag)


    # Get the node at the specified location. May return None.    
    def get_node( self, nodelocation : NodeLocation ):
        for node in self.nodes:
            if node.position == nodelocation.position and node.is_head == nodelocation.is_head and node.layer == nodelocation.layer and node.num == nodelocation.num:
                return node

        return None


    # Get first node with specified tags. May return None.    
    def get_node_by_tag( self, major_tag : str, minor_tag : str):
        for node in self.nodes:
            if node.contains_tag( major_tag, minor_tag ):
                return node

        return None
    

    # Add the tag to the node location (creating a node if necessary). Returns number of tags added as 0 or 1
    def add_node_tag( self, nodelocation, major_tag : str, minor_tag : str ):

        the_node = self.get_node( nodelocation )
        if the_node == None:

            the_node = UsefulNode(nodelocation.position, nodelocation.layer, nodelocation.is_head, nodelocation.num, [])

            self.nodes += [the_node]

        return the_node.add_tag(major_tag, minor_tag)


    # Sort the nodes into position, layer, is_head, num order
    def sort_nodes(self):
        # We use 2-digit position numbers so that they sort correctly
        self.nodes = sorted(self.nodes, key=lambda obj: (obj.name(False)))


    # Save the nodes and tags to a json file
    def save_nodes(self, filename, major_tag = ""):
        dict_list = [node.to_dict(major_tag) for node in self.nodes]
        with open(filename, 'w') as file:
            json.dump(dict_list, file, default=lambda o: o.__dict__)
    
            
    # Load the nodes and tags from a json file. (Does not delete existing nodes.)
    def load_nodes(self, filename):
        with open(filename, 'r') as file:
            dict_list = json.load(file)
            for data in dict_list:
                data_node = UsefulNode.from_dict(data)
                
                # Find/create the corresponding in-memory node
                the_node = self.get_node( data_node )
                if the_node == None:
                    the_node = UsefulNode(data_node.position, data_node.layer, data_node.is_head, data_node.num, [])
                    self.nodes += [the_node]

                # Load/merge the tags
                for tag in data_node.tags:
                    major_tag, minor_tag = tag.split(":")
                    the_node.add_tag(major_tag, minor_tag)