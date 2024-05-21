from .quanta_constants import QType, QCondition


# Either the node has or doesnt have matching tag(s). Return the matching tag(s) if any
def get_quanta_binary(_, node, major_tag : str, minor_tag : str, __):

    cell_text = ""

    node_tags = node.filter_tags( major_tag, minor_tag )
    for tag in node_tags:
        cell_text += tag + " "

    color_index = 0 if cell_text == "" else 1

    return cell_text, color_index


# Map ALL useful nodes. If node has a QType.ALGO tag, return the tag(s)
def get_quanta_algo(_, node, __ : str, minor_tag : str, ___):

    cell_text = ""

    node_tags = node.filter_tags( QType.ALGO.value, minor_tag )
    for tag in node_tags:
        cell_text += tag + " "

    color_index = 1 if cell_text == "" else 2

    if cell_text == "":
        # Ensure all cells are shown    
        cell_text = " "

    return cell_text, color_index