from .quanta_constants import MAX_ATTENTION_TAGS, MIN_ATTENTION_PERC 
from .useful_node import position_name_to_int 


# Return the token_position_meanings that this node (attention head) pays attention to
def get_quanta_attention(cfg, node, major_tag : str, minor_tag : str, num_shades : int):
    cell_text = ""
    color_index = 0

    if node.is_head:
        sum_perc = 0
        node_tags = node.filter_tags( major_tag )
        for minor_tag in node_tags:
            node_parts = minor_tag.split("=")
            token_pos = position_name_to_int(node_parts[0])
            the_perc = int(node_parts[1])
            if the_perc > MIN_ATTENTION_PERC:
                cell_text += cfg.token_position_meanings[token_pos] + " "
                sum_perc += the_perc

        cell_text = cell_text.rstrip(" ")
        color_index = num_shades - sum_perc // num_shades    # Want >90% => Dark-Green, and <10% => Yellow

        if len(node_tags) == MAX_ATTENTION_TAGS:
            # Number of input tokens that node attended to could be > MAX_ATTENTION_TAGS so show yellow
            color_index = num_shades-1

    return cell_text, color_index
