

# Return the percentage of questions that the model failed to predict when this node was ablated.
# If the node fails on say 3 of 1000 questions, rather than return 0%, for clarity we show <1%
def get_quanta_fail_perc( _, node, major_tag, __, num_shades):
    cell_text = node.only_tag( major_tag )
    value = int(cell_text) if cell_text != "" else 0

    color_index = value // num_shades
    cell_text = (str(value) if value > 0 else "<1")

    return cell_text, color_index
