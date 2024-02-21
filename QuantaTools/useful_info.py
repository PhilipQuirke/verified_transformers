from .useful_node import UsefulNode


class UsefulInfo():
  # sparce ordered list of useful token positions e.g. 0,1,8,9,10,11
  positions = []

  # sparce ordered list of useful rows e.g. 0,1,4 representing L0H0, L0H1 and L0MLP
  rows = []

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


  # Add a quanta row that we know is used in calculations
  def add_useful_row(self, row):
    if not (row in self.rows):
      self.rows += [row]


  def print__tags(self):
    for node in self.nodes:
      print( node.name(), node.tags )



  def reset_tags( self, major_tag = "" ):
    for node in self.nodes:
      node.reset_tags(major_tag)


  def get_node( self, the_position, the_layer, is_head, the_num ):
    for node in self.nodes:
      if node.position == the_position and node.is_head == is_head and node.layer == the_layer and node.num == the_num:
        return node

    return None


  def add_node_tag( self, the_position, the_layer, is_head, the_num, major_tag, minor_tag ):

    the_node = self.get_node( the_position, the_layer, is_head, the_num )
    if the_node == None:

      the_node = UsefulNode(the_position, the_layer, is_head, the_num, [])

      self.nodes += [the_node]

    the_node.add_tag(major_tag, minor_tag)


  def sort_nodes(self):
    self.nodes = sorted(self.nodes, key=lambda obj: (obj.position, obj.layer, obj.is_head, obj.num))


  def save_nodes(self, filename):
    dict_list = [node.to_dict() for node in self.nodes]
    with open(filename, 'w') as file:
        json.dump(dict_list, file, default=lambda o: o.__dict__)


  # Filter by a set of [QuantaTools.Filter, major_tag, minor_tag ]
  # filter_heads(self, [[MUST, POSITION_MAJOR_TAG, P14], [MUST, IMPACT_MAJOR_TAG, A543], [NOT, ALGO_MAJOR_TAG, D2.BA] ])
  def filter_heads(self, filters):
    answer = []
    for node in self.nodes:
      if node.is_head:
        include = True
        for filter in filters:
          quanta_filter = filter[0]
          assert isinstance(quanta_filter, QuantaFilter)
          major_tag = filter[1]
          minor_tag = filter[2]
          if major_tag == POSITION_MAJOR_TAG:
            if quanta_filter == QuantaFilter.MUST:
              include &= (position_name(node.position) == minor_tag)
            elif quanta_filter == QuantaFilter.NOT:
              include &= (not position_name(node.position) == minor_tag)
            elif quanta_filter == QuantaFilter.MUSTMAY:
              include &= True     # No effect
          else:
            if quanta_filter == QuantaFilter.MUST or quanta_filter == QuantaFilter.CONTAINS:
              include &= node.contains_tag(major_tag,minor_tag)
            elif quanta_filter == QuantaFilter.NOT:
              include &= not node.contains_tag(major_tag,minor_tag)
            elif quanta_filter == QuantaFilter.MAY:
              include &= True     # No effect
        if include:
          answer += [node]

    return answer


useful_info = UsefulInfo()
