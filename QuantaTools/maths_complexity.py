import torch
import transformer_lens.utils as utils

from .quanta_type import QuantaType
from .maths_utilities import tokens_to_unsigned_int
from .maths_vocab import MathsTokens
from .maths_tag import MathsBehavior


# Analyse and return the question complexity for the Addition (S0 to S4) or Subtraction (M0 to NG) questions
def get_maths_question_complexity(cfg, question):
    qlist = utils.to_numpy(question)
    inputs = qlist[:cfg.num_question_positions]
    operator = qlist[cfg.n_digits]

    if operator == MathsTokens.PLUS:

        # Locate the MC and MS digits (if any)
        mc = torch.zeros(cfg.n_digits).to(torch.int64)
        ms = torch.zeros(cfg.n_digits).to(torch.int64)
        for dn in range(cfg.n_digits):
            if inputs[dn] + inputs[dn + cfg.n_digits + 1] == 9:
                ms[cfg.n_digits-1-dn] = 1
            if inputs[dn] + inputs[dn + cfg.n_digits +1] > 9:
                mc[cfg.n_digits-1-dn] = 1

        if torch.sum(mc) == 0:
            return QuantaType.MATH_ADD, MathsBehavior.ADD_S0_TAG

        if torch.sum(ms) == 0:
            return QuantaType.MATH_ADD, MathsBehavior.ADD_S1_TAG

        for dn in range(cfg.n_digits-4):
            if mc[dn] == 1 and ms[dn+1] == 1 and ms[dn+2] == 1 and ms[dn+3] == 1 and ms[dn+4] == 1:
                return QuantaType.MATH_ADD, MathsBehavior.ADD_S5_TAG # MC cascades 4 or more digits

        for dn in range(cfg.n_digits-3):
            if mc[dn] == 1 and ms[dn+1] == 1 and ms[dn+2] == 1 and ms[dn+3] == 1:
                return QuantaType.MATH_ADD, MathsBehavior.ADD_S4_TAG # MC cascades 3 or more digits

        for dn in range(cfg.n_digits-2):
            if mc[dn] == 1 and ms[dn+1] == 1 and ms[dn+2] == 1:
                return QuantaType.MATH_ADD, MathsBehavior.ADD_S3_TAG # MC cascades 2 or more digits

        for dn in range(cfg.n_digits-1):
            if mc[dn] == 1 and ms[dn+1] == 1:
                return QuantaType.MATH_ADD, MathsBehavior.ADD_S2_TAG # Simple US 9

        return QuantaType.MATH_ADD, MathsBehavior.ADD_S1_TAG


    if operator == MathsTokens.MINUS:
        a = tokens_to_unsigned_int( question, 0, cfg.n_digits )
        b = tokens_to_unsigned_int( question, cfg.n_digits + 1, cfg.n_digits )
        if a - b < 0:
            return QuantaType.MATH_SUB, MathsBehavior.SUB_NG_TAG

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
            return QuantaType.MATH_SUB, MathsBehavior.SUB_S0_TAG

        # Evaluate subtraction "cascade multiple steps" questions
        for dn in range(cfg.n_digits-3):
            if bo[dn] == 1 and mz[dn+1] == 1 and mz[dn+2] == 1 and mz[dn+3] == 1:
                return QuantaType.MATH_SUB, "M4+" # BO cascades 3 or more digits

        # Evaluate subtraction "cascade multiple steps" questions
        for dn in range(cfg.n_digits-2):
            if bo[dn] == 1 and mz[dn+1] == 1 and mz[dn+2] == 1:
                return QuantaType.MATH_SUB, MathsBehavior.SUB_S3_TAG # BO cascades 2 or more digits

        # Evaluate subtraction "cascade 1" questions
        for dn in range(cfg.n_digits-1):
            if bo[dn] == 1 and mz[dn+1] == 1:
                return QuantaType.MATH_SUB, MathsBehavior.SUB_S2_TAG # BO cascades 1 digit

        return QuantaType.MATH_SUB, MathsBehavior.SUB_S1_TAG


    # Should never get here
    print("get_question_complexity OP? exception", question)
    return "", "OP?"


# Analyze the tags associated with node, to show the minimum mathematical complexity
# That is, what is the simpliest type of question that this node is needed for?
def get_maths_min_complexity(_, node, major_tag, minor_tag, num_shades):
    color_index = 0
    cell_text = node.min_tag_suffix( major_tag, minor_tag )
    if cell_text != "" :
        cell_text = cell_text[0:2]
        color_index = int(cell_text[1]) if len(cell_text) > 1 and cell_text[1].isdigit() else num_shades-1

    return cell_text, color_index