import json

from .quanta_filter import QuantaFilter
from .quanta_type import QuantaType
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, NodeLocation, UsefulNode 


# convert 3 to "A3"
def answer_name(n):
  return "A" + str(n)
  

class UsefulInfo():
  # The number of "question" (input) token positions 
  num_question_positions : int = 12
  
  # The number of "answer" (output) token positions 
  num_answer_positions : int = 6
  
  # Do we name the answer tokens as A5, A4, A3, A2, A1, A0 or A0, A1, A2, A3, A4, A5?
  answer_meanings_ascend : bool = True
  
  # List of (short) strings representing the meaning of each token position.
  # For example D5, D4, D3, D2, D1, D0, +, D'5, D'4, D'3, D'2, D'1, D'0, =, A6, A5, A4, A3, A2, A1, A0
  # Used in node tag, in the column headings of quanta-maps, etc. 
  token_position_meanings = []

  # sparce ordered list of useful (question and answer) token positions e.g. 0,1,8,9,10,11
  positions = []

  # list of useful (attention head and MLP neuron) nodes
  nodes = []

  
  def initialize_token_positions(self, num_question_positions, num_answer_positions, answer_meanings_ascend ):
    self.num_question_positions = num_question_positions
    self.num_answer_positions = num_answer_positions
    self.answer_meanings_ascend = answer_meanings_ascend   
    self.default_token_position_meanings()

  
  # Default list of strings representing the token positions meanings
  def default_token_position_meanings(self):
    self.token_position_meanings = []
    for i in range(self.num_question_positions):
      self.token_position_meanings += ["P"+str(i)]
    for i in range(self.num_answer_positions):
      self.token_position_meanings += [answer_name(i if self.answer_meanings_ascend else self.num_answer_positions - i - 1 )]
      
  
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
