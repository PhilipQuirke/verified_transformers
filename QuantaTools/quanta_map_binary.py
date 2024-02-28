from .quanta_filter import QuantaFilter
from .quanta_type import QuantaType
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, NodeLocation, UsefulNode 
from .useful_info import UsefulInfo, useful_info


# Either the node has or doesnt have matching tag(s). Show the tag(s) or show "??"
def get_quanta_binary(node, major_tag, minor_tag, shades):

  cell_text = ""
  color_index = 0

  node_tags = node.filter_tags( major_tag, minor_tag )
  for tag in node_tags:
    cell_text += tag + " "

  if cell_text == "":
    cell_text = "??"
  else:
    color_index = 1

  return cell_text, color_index
