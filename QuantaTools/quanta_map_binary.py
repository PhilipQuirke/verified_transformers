

# Either the node has or doesnt have matching tag(s). Return the matching tag(s) if any
def get_quanta_binary(_, node, major_tag, minor_tag, __):

    cell_text = ""

    node_tags = node.filter_tags( major_tag, minor_tag )
    for tag in node_tags:
        cell_text += tag + " "

    color_index = 0 if cell_text == "" else 1

    return cell_text, color_index
