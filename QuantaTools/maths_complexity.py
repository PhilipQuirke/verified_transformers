import matplotlib.pyplot as plt
import torch
import transformer_lens.utils as utils

from .useful_node import position_name
from .quanta_constants import QType, FAIL_SHADES, ATTN_SHADES, ALGO_SHADES, MATH_ADD_SHADES, MATH_SUB_SHADES
from .quanta_filter import FilterPosition, filter_nodes
from .quanta_map import create_colormap, pale_color
from .quanta_map_attention import get_quanta_attention
from .quanta_map_failperc import get_quanta_fail_perc
from .quanta_map_binary import get_quanta_binary
from .quanta_map_impact import get_quanta_impact
from .maths_utilities import tokens_to_unsigned_int
from .maths_constants import MathsToken, MathsBehavior


# Analyse and return the question complexity for the Addition (S0 to S4) or Subtraction (M0 to NG) questions
def get_maths_question_complexity(cfg, question):
    qlist = utils.to_numpy(question)
    inputs = qlist[:cfg.num_question_positions]
    operator = qlist[cfg.n_digits]

    if operator == MathsToken.PLUS:

        # Locate the MC and MS digits (if any)
        mc = torch.zeros(cfg.n_digits).to(torch.int64)
        ms = torch.zeros(cfg.n_digits).to(torch.int64)
        for dn in range(cfg.n_digits):
            if inputs[dn] + inputs[dn + cfg.n_digits + 1] == 9:
                ms[cfg.n_digits-1-dn] = 1
            if inputs[dn] + inputs[dn + cfg.n_digits +1] > 9:
                mc[cfg.n_digits-1-dn] = 1

        if torch.sum(mc) == 0:
            return QType.MATH_ADD, MathsBehavior.ADD_S0_TAG

        if torch.sum(ms) == 0:
            return QType.MATH_ADD, MathsBehavior.ADD_S1_TAG

        for dn in range(cfg.n_digits-4):
            if mc[dn] == 1 and ms[dn+1] == 1 and ms[dn+2] == 1 and ms[dn+3] == 1 and ms[dn+4] == 1:
                return QType.MATH_ADD, MathsBehavior.ADD_S5_TAG # MC cascades 4 or more digits

        for dn in range(cfg.n_digits-3):
            if mc[dn] == 1 and ms[dn+1] == 1 and ms[dn+2] == 1 and ms[dn+3] == 1:
                return QType.MATH_ADD, MathsBehavior.ADD_S4_TAG # MC cascades 3 or more digits

        for dn in range(cfg.n_digits-2):
            if mc[dn] == 1 and ms[dn+1] == 1 and ms[dn+2] == 1:
                return QType.MATH_ADD, MathsBehavior.ADD_S3_TAG # MC cascades 2 or more digits

        for dn in range(cfg.n_digits-1):
            if mc[dn] == 1 and ms[dn+1] == 1:
                return QType.MATH_ADD, MathsBehavior.ADD_S2_TAG # Simple US 9

        return QType.MATH_ADD, MathsBehavior.ADD_S1_TAG


    if operator == MathsToken.MINUS:
        a = tokens_to_unsigned_int( question, 0, cfg.n_digits )
        b = tokens_to_unsigned_int( question, cfg.n_digits + 1, cfg.n_digits )
        if a - b < 0:
            return QType.MATH_SUB, MathsBehavior.SUB_NG_TAG

        # Locate the BO and MZ digits (if any)
        bo = torch.zeros(cfg.n_digits).to(torch.int64)
        mz = torch.zeros(cfg.n_digits).to(torch.int64)
        for dn in range(cfg.n_digits):
            if inputs[dn] - inputs[dn + cfg.n_digits + 1] < 0:
                bo[cfg.n_digits-1-dn] = 1
            if inputs[dn] - inputs[dn + cfg.n_digits +1] == 0:
                mz[cfg.n_digits-1-dn] = 1

        # Evaluate BaseSub questions - when no column generates a Borrow One
        if torch.sum(bo) == 0:
            return QType.MATH_SUB, MathsBehavior.SUB_S0_TAG

        # Evaluate subtraction "cascade multiple steps" questions
        for dn in range(cfg.n_digits-3):
            if bo[dn] == 1 and mz[dn+1] == 1 and mz[dn+2] == 1 and mz[dn+3] == 1:
                return QType.MATH_SUB, "M4+" # BO cascades 3 or more digits

        # Evaluate subtraction "cascade multiple steps" questions
        for dn in range(cfg.n_digits-2):
            if bo[dn] == 1 and mz[dn+1] == 1 and mz[dn+2] == 1:
                return QType.MATH_SUB, MathsBehavior.SUB_S3_TAG # BO cascades 2 or more digits

        # Evaluate subtraction "cascade 1" questions
        for dn in range(cfg.n_digits-1):
            if bo[dn] == 1 and mz[dn+1] == 1:
                return QType.MATH_SUB, MathsBehavior.SUB_S2_TAG # BO cascades 1 digit

        return QType.MATH_SUB, MathsBehavior.SUB_S1_TAG


    # Should never get here
    print("get_question_complexity OP? exception", question)
    return QType.UNKNOWN, MathsBehavior.UNKNOWN


# Analyze the tags associated with node, to show the minimum mathematical complexity
# That is, what is the simpliest type of question that this node is needed for?
def get_maths_min_complexity(_, node, major_tag, minor_tag, num_shades):
    color_index = 0
    cell_text = node.min_tag_suffix( major_tag, minor_tag )
    if cell_text != "" :
        cell_text = cell_text[0:2]
        color_index = int(cell_text[1]) if len(cell_text) > 1 and cell_text[1].isdigit() else num_shades-1

    return cell_text, color_index



# Calculate a table of the known quanta for the specified position for each useful node
def calc_maths_quanta_for_position_nodes(cfg, position):

    columns = ["Posn meaning", "Node name", "Answer impact", "Algo purpose", "Attends to", "Min Add Complex", "Min Sub Complex", "Fail %"]
    text_data = None
    shade_data = None

    nodelist = filter_nodes(cfg.useful_nodes, FilterPosition(position_name(position)))
    for node in nodelist.nodes:
        position_meaning = cfg.token_position_meanings[position]
        node_name = node.name()
        node_impact, impact_shade = get_quanta_impact( cfg, node, QType.IMPACT.value, "", cfg.num_answer_positions )
        node_algorithm_purpose, algo_shade = get_quanta_binary( cfg, node, QType.ALGO.value, "", ALGO_SHADES)
        node_attention, attention_shade = get_quanta_attention( cfg, node, QType.ATTN.value, "", ATTN_SHADES )
        node_add_complexity, add_complexity_shade = get_maths_min_complexity( cfg, node, QType.MATH_ADD.value, "", MATH_ADD_SHADES)
        node_sub_complexity, sub_complexity_shade = get_maths_min_complexity( cfg, node, QType.MATH_SUB.value, "", MATH_SUB_SHADES)
        node_fail_perc, fail_perc_shade = get_quanta_fail_perc( cfg, node, QType.FAIL.value, "", FAIL_SHADES)

        shade_array = [0, 0, 
            1.0 * impact_shade / cfg.num_answer_positions, 
            1.0 * algo_shade / ALGO_SHADES, 
            1.0 * attention_shade / ATTN_SHADES, 
            1.0 * add_complexity_shade / MATH_ADD_SHADES, 
            1.0 * sub_complexity_shade / MATH_SUB_SHADES, 
            1.0 * fail_perc_shade / FAIL_SHADES]
        if shade_data is None:
            shade_data = [shade_array]            
        else:
            shade_data += [shade_array]

        text_array = [position_meaning, node_name, node_impact, node_algorithm_purpose, node_attention, node_add_complexity, node_sub_complexity, node_fail_perc]
        if text_data is None:
            text_data = [text_array]
        else:
            text_data += [text_array]
            

    if not text_data is None:
        _, ax = plt.subplots(figsize=(16,2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=text_data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)  # Set the font size here
        table.scale(1, 1.5)  # The first parameter scales column widths, the second scales row heights

        # Set column headings to bold
        for col, _ in enumerate(columns):
            table[(0, col)].get_text().set_weight('bold')

        standard_map = create_colormap( True ) # Light green color
        specific_map = create_colormap( False ) # Light blue color

        # Color all non-blank body cells in the specified column a shade of green or blue
        for row in range(len(text_data)):
            for col in range(2, len(columns)):
                if text_data[row][col] != "":
                    the_color_map = specific_map if col <= 4 or col == 7 else standard_map 
                    table[(row+1, col)].set_facecolor(pale_color(the_color_map(shade_data[row][col])))




