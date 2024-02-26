import json

from .quanta_filter import QuantaFilter
from .quanta_type import QuantaType
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, NodeLocation, UsefulNode 


class UsefulInfo():
  # sparce ordered list of useful token positions e.g. 0,1,8,9,10,11
  positions = []

  # list of useful nodes
  nodes = []


  def min_useful_position(self):
    return min(self.positions)


  def max_useful_position(self):
    return max(self.positions)


  # Add a token position that we know is used in calculations
  def add_useful_position(self, position):
    if not (position in self.positions):
      self.positions += [position]


  def print_node_tags(self, major_tag = ""):
    for node in self.nodes:
      if major_tag == "":
        print( node.name(), node.tags )
      else:
        print( node.name(), node.filter_tags(major_tag) )       


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


  # Filter by a list of [QuantaFilter, QuantaType, minor_tag] clauses
  # If there are multiple QuantaType.ATTENTION clauses, the head only has to satisft one clause. That is these clauses are OR'ed.
  # filter_heads(self, [[QuantaFilter.MUST, QuantaType.POSITION, "P14"], [QuantaFilter.MUST, QuantaType.IMPACT, "A543"], [QuantaFilter.NOT, QuantaType.ALGO, "D2.BA"] ])
  def filter_heads(self, the_filters):
    answer = []
    for node in self.nodes:
      if node.is_head:
        include = True
        attention_clause_found = False
        attention_match_found = False   
        
        for a_filter in the_filters:
          quanta_filter = a_filter[0]
          assert isinstance(quanta_filter, QuantaFilter)
          major_tag = a_filter[1]
          minor_tag = a_filter[2]
          
          if major_tag == QuantaType.POSITION:
            if quanta_filter == QuantaFilter.MUST:
              include &= (position_name(node.position) == minor_tag)
            elif quanta_filter == QuantaFilter.NOT:
              include &= (not position_name(node.position) == minor_tag)
            elif quanta_filter == QuantaFilter.MAY:
              pass
              
          elif major_tag == QuantaType.ATTENTION:
            # A node can pay attention to several tokens. The filters can name multiple input tokens which are treated as an OR statement
            attention_clause_found = True
            if quanta_filter == QuantaFilter.MUST or quanta_filter == QuantaFilter.CONTAINS:
              attention_match_found = attention_match_found or node.contains_tag(major_tag,minor_tag)
            elif quanta_filter == QuantaFilter.NOT:
              attention_match_found = attention_match_found or not node.contains_tag(major_tag,minor_tag)
            elif quanta_filter == QuantaFilter.MAY:
              pass     
              
          else:
            if quanta_filter == QuantaFilter.MUST or quanta_filter == QuantaFilter.CONTAINS:
              include &= node.contains_tag(major_tag,minor_tag)
            elif quanta_filter == QuantaFilter.NOT:
              include &= not node.contains_tag(major_tag,minor_tag)
            elif quanta_filter == QuantaFilter.MAY:
              pass

        if attention_clause_found and not attention_match_found:
          include = False
        
        if include:
          answer += [node]

    return answer


useful_info = UsefulInfo()
