from .quanta_constants import MAX_ATTN_TAGS, MIN_ATTN_PERC, ATTN_ORDER_DIFF, ATTN_ORDER_MIN
from .useful_node import position_name_to_int 


# Return the token_position_meanings that this node (attention head) pays attention to
def get_quanta_attention(cfg, node, major_tag : str, minor_tag : str, num_shades : int):
    cell_text = ""
    color_index = 0

    if node.is_head:
        node_tags = node.filter_tags( major_tag )
        tags_with_perc = []  # To store tags and their percentages
        
        for minor_tag in node_tags:
            node_parts = minor_tag.split("=")
            token_pos = position_name_to_int(node_parts[0])
            the_perc = int(node_parts[1])
            if the_perc > MIN_ATTN_PERC:
                tags_with_perc.append((cfg.token_position_meanings[token_pos], the_perc))
                
        # Now process the collected tags
        if len(tags_with_perc) >= 2:
            # Sort primarily by percentage (descending), secondary by name (ascending) if percentages are close
            tags_with_perc.sort(key=lambda x: (-x[1], x[0]))
    
            # Check if the top two percentages are within 5% of each other, or if both are >= 40%
            tag_0_1 = tags_with_perc[0][1] 
            tag_1_1 = tags_with_perc[1][1]
            if (abs(tag_0_1 - tag_1_1) <= ATTN_ORDER_DIFF) or ((tag_0_1 >= ATTN_ORDER_MIN) and (tag_1_1 >= ATTN_ORDER_MIN)):
                # If so, sort these two alphabetically
                sorted_by_name = sorted(tags_with_perc[:2], key=lambda x: x[0])
                tags_with_perc[:2] = sorted_by_name


        # Generate the cell_text
        cell_text = ""
        sum_perc = 0
        for tag, perc in tags_with_perc:
            cell_text += tag + " "
            sum_perc += perc
        cell_text = cell_text.rstrip(" ")
        color_index = num_shades - sum_perc // num_shades    # Want >90% => Dark-Green, and <10% => Yellow

        if len(node_tags) == MAX_ATTN_TAGS:
            # Number of input tokens that node attended to could be > MAX_ATTN_TAGS so show yellow
            color_index = num_shades-1

    return cell_text, color_index
