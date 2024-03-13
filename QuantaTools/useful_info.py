import matplotlib.pyplot as plt

from .quanta_filter import QuantaFilter
from .quanta_type import QuantaType
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, answer_name, NodeLocation, UsefulNode, UsefulNodeList


class UsefulInfo(UsefulNodeList):
  
  # Vocabulary: Map from each character to each token
  char_to_token : dict = { }

  # The number of "question" (input) token positions e.g. len("12340+12340=")
  num_question_positions : int = 12
  
  # The number of "answer" (output) token positions  e.g. len("+024680") 
  num_answer_positions : int = 7
  
  # Do we name the answer tokens as A5, A4, A3, A2, A1, A0 or A0, A1, A2, A3, A4, A5?
  answer_meanings_ascend : bool = True
  
  # List of (short) strings representing the meaning of each token position.
  # For example D5, D4, D3, D2, D1, D0, +, D'5, D'4, D'3, D'2, D'1, D'0, =, A6, A5, A4, A3, A2, A1, A0
  # Used in node tag, in the column headings of quanta-maps, etc. 
  token_position_meanings = []

  # sparce ordered list of useful (question and answer) token positions e.g. 0,1,8,9,10,11
  positions = []

  
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


  # Show the positions, their meanings, and the number of questions that failed when that position is ablated in a 3 row table
  def calc_position_failures_map(self, num_failures_list, width_inches=16):
    columns = ["Posn"]
    for i in range(len(self.token_position_meanings)):
      columns += [position_name(i)]
    
    rows = ["Posn", "# fails"]
    data = [
        ["Posn"] + self.token_position_meanings,
        ["# fails"] + num_failures_list
    ]
    
    fig, ax = plt.subplots(figsize=(width_inches,1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Set the font size here
    table.scale(1, 1.5)  # The first parameter scales column widths, the second scales row heights


useful_info = UsefulInfo()
