import json

from .quanta_filter import QuantaFilter
from .quanta_type import QuantaType
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, NodeLocation, UsefulNode 


class UsefulInfo():
  # sparce ordered list of useful token positions e.g. 0,1,8,9,10,11
  positions = []

  # list of useful nodes
  nodes = []

  # list of strings representing the meaning of each token position.
  # Used in the column headings of quanta-maps.
  # Use default_token_position_meanings() to default list to P0, P1, etc. 
  # Or set list to say D5, D4, D3, D2, D1, D0, +, D'5, D'4, D'3, D'2, D'1, D'0, =, A6, A5, A4, A3, A2, A1, A0
  token_position_meanings = []


  # Default list of strings used to in the column headings of quanta-maps
  def default_token_position_meanings(num_positions):
    token_position_meanings = []
    for i in range(num_positions):
      token_position_meanings += ["P"+str(i)]
  
  
  def min_useful_position(self):
    return min(self.positions)


  def max_useful_position(self):
    return max(self.positions)


  # Add a token position that we know is used in calculations
  def add_useful_position(self, position):
    if not (position in self.positions):
      self.positions += [position]


  def print_node_tags(self, major_tag = "", show_empty_tags = True):
    for node in self.nodes:
      tags = node.tags if major_tag == "" else node.filter_tags(major_tag)
      if show_empty_tags or len(tags) > 0 :
        print( node.name(), tags )       


  def reset_node_tags( self, major_tag = "" ):
    for node in self.nodes:
      node.reset_tags(major_tag)


  def get_node( self, nodelocation ):
    for node in self.nodes:
      if node.position == nodelocation.position and node.is_head == nodelocation.is_head and node.layer == nodelocation.layer and node.num == nodelocation.num:
        return node

    return None


  def add_node_tag( self, nodelocation, major_tag, minor_tag ):

    the_node = self.get_node( nodelocation )
    if the_node == None:

      the_node = UsefulNode(nodelocation.position, nodelocation.layer, nodelocation.is_head, nodelocation.num)

      self.nodes += [the_node]

    the_node.add_tag(major_tag, minor_tag)


  def sort_nodes(self):
    self.nodes = sorted(self.nodes, key=lambda obj: (obj.position, obj.layer, obj.is_head, obj.num))


  def save_nodes(self, filename):
    dict_list = [node.to_dict() for node in self.nodes]
    with open(filename, 'w') as file:
        json.dump(dict_list, file, default=lambda o: o.__dict__)


useful_info = UsefulInfo()
