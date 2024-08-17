import matplotlib.pyplot as plt

from QuantaTools.model_token_to_char import token_to_char
from QuantaTools.model_pca import calc_pca_for_an, pca_evr_0_percent
from QuantaTools.quanta_constants import QType
from QuantaTools.quanta_file_utils import save_plt_to_file
from QuantaTools.useful_node import answer_name, NodeLocation
from .maths_constants import MathsBehavior, MathsToken
from .maths_test_questions import EACH_CASE_TRICASE_QUESTIONS as TRICASE_QUESTIONS



def plot_pca_for_an(ax, pca_attn_outputs, title, num_questions=TRICASE_QUESTIONS):
    """
    Plot the PCA of PnLnHn's attention pattern, using ST8, ST9, ST10 questions that differ in the An digit
    Assumes that we have equal number of questions for each tricase.
    """
    ax.scatter(pca_attn_outputs[:num_questions, 0], pca_attn_outputs[:num_questions, 1], color='red', label='ST8 (0-8)') # st8 questions
    ax.scatter(pca_attn_outputs[num_questions:2*num_questions, 0], pca_attn_outputs[num_questions:2*num_questions, 1], color='green', label='ST9') # st9 questions
    ax.scatter(pca_attn_outputs[2*num_questions:, 0], pca_attn_outputs[2*num_questions:, 1], color='blue', label='ST10 (10-18)') # st10 questions
    if title != "" :
        ax.set_title(title)


def pca_op_tag(the_digit, operation):
    minor_tag_prefix = MathsBehavior.ADD_PCA_TAG if operation == MathsToken.PLUS else MathsBehavior.SUB_PCA_TAG
    return answer_name(the_digit)  + "." + minor_tag_prefix.value


def _build_title_and_error_message(cfg, node_location, operation, answer_digit):
    title = node_location.name() + ' A' + str(answer_digit)
    error_message = ("calc_pca_for_an Failed:" + node_location.name() + " " +
                     token_to_char(cfg, operation) + " " + answer_name(answer_digit))
    return title, error_message


def manual_node_pca(cfg, ax, position, layer, num, operation, answer_digit):
    node_location = NodeLocation(position, layer, True, num)
    test_inputs = cfg.tricase_questions_dict[(answer_digit, operation)]

    title, error_message = _build_title_and_error_message(
        cfg=cfg, node_location=node_location, operation=operation, answer_digit=answer_digit
    )
    pca, pca_attn_outputs, title, _ = calc_pca_for_an(
        cfg=cfg, node_location=node_location, test_inputs=test_inputs, title=title, error_message=error_message
    )
    
    if pca_attn_outputs is None:
        return

    plot_pca_for_an(ax, pca_attn_outputs, title)

    major_tag = QType.MATH_ADD if operation == MathsToken.PLUS else QType.MATH_SUB # Does not handle NEG case
    cfg.add_useful_node_tag( node_location, major_tag.value, pca_op_tag(answer_digit, operation) )



def plot_nodes_pca_start(nodes):
    
    n_cols = 4
    n_rows = 1 + (len(nodes)+1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols)
    fig.set_figheight(n_rows*2 + 1)
    fig.set_figwidth(10)
    
    return n_cols, n_rows, fig, axs


def plot_nodes_pca_end(n_cols, n_rows, axs, cfg, title, index):

    # Do we have room to add the legend as a plot area?
    if index + 1 < n_rows * n_cols:
        index += 1
        lines_labels = [axs[0,0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        axs[index // n_cols, index % n_cols].legend(lines, labels)
        axs[index // n_cols, index % n_cols].axis('off') # Now, to hide the last subplot

    # Remove any graphs we dont need
    while index + 1 < n_rows * n_cols:
        index += 1
        ax = axs[index // n_cols, index % n_cols]
        ax.remove()
        
    plt.tight_layout()
    save_plt_to_file(cfg=cfg, full_title=title)
    plt.show()    


# Plot the PCA diagram for a list of manually provided nodes
def manual_nodes_pca(cfg, operation, nodes):
    
    print("Manual PCA tags for", cfg.model_name, "with operation", token_to_char(cfg, operation))
    title = cfg.model_name + "_PCA_" + token_to_char(cfg, operation)

    n_cols, n_rows, fig, axs = plot_nodes_pca_start(nodes)

    index = 0
    for node in nodes:
        manual_node_pca(cfg=cfg, ax=axs[index // n_cols, index % n_cols], position=node[0],
                        layer=node[1], num=node[2], operation=operation, answer_digit=node[3])
        index += 1

    plot_nodes_pca_end(n_cols, n_rows, axs, cfg, title, index)

