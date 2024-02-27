from .quanta_filter import QuantaFilter
from .quanta_type import QuantaType
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, NodeLocation, UsefulNode 
from .useful_info import UsefulInfo, useful_info


def get_quanta_fail_perc( node, major_tag, minor_tag, shades):
  cell_text = node.only_tag( major_tag )
  value = int(cell_text) if cell_text != "" else 0

  if value == 100:
    value = 99 # Avoid overlapping figures in the matrix.
  color_index = value // shades
  cell_text = (str(value) if value > 0 else "<1") + "%"

  return cell_text, color_index
