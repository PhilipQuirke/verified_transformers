import matplotlib.pyplot as plt

from QuantaTools.model_token_to_char import token_to_char
from QuantaTools.model_pca import calc_pca_for_an, pca_evr_0_percent
from QuantaTools.quanta_constants import QType
from QuantaTools.quanta_file_utils import save_plt_to_file
from QuantaTools.useful_node import answer_name, NodeLocation
from .maths_constants import MathsBehavior, MathsToken
from .maths_test_questions import TRICASE_QUESTIONS



# Plot the PCA of PnLnHn's attention pattern, using T8, T9, T10 questions that differ in the An digit
def plot_pca_for_an(ax, pca_attn_outputs, title):
    ax.scatter(pca_attn_outputs[:TRICASE_QUESTIONS, 0], pca_attn_outputs[:TRICASE_QUESTIONS, 1], color='red', label='T8 (0-8)') # t8 questions
    ax.scatter(pca_attn_outputs[TRICASE_QUESTIONS:2*TRICASE_QUESTIONS, 0], pca_attn_outputs[TRICASE_QUESTIONS:2*TRICASE_QUESTIONS, 1], color='green', label='T9') # t9 questions
    ax.scatter(pca_attn_outputs[2*TRICASE_QUESTIONS:, 0], pca_attn_outputs[2*TRICASE_QUESTIONS:, 1], color='blue', label='T10 (10-18)') # t10 questions
    if title != "" :
        ax.set_title(title)

def pca_op_tag(the_digit, operation, strong):
    minor_tag_prefix = MathsBehavior.ADD_PCA_TAG if operation == MathsToken.PLUS else MathsBehavior.SUB_PCA_TAG
    return answer_name(the_digit)  + "." + minor_tag_prefix.value + ( "" if strong else ".Weak")

# Use a different plotting function if you need.
def manual_node_pca(cfg, ax, position, layer, num, operation, answer_digit, plotting_function=plot_pca_for_an):
    node_location = NodeLocation(position, layer, True, num)
    test_inputs = cfg.tricase_questions_dict[(answer_digit, operation)]
    title = node_location.name() + ', A' + str(answer_digit)
    error_message = ("calc_pca_for_an Failed:" + node_location.name() + " " +
                     token_to_char(cfg, operation) + " " + answer_name(answer_digit))

    pca, pca_attn_outputs, title = calc_pca_for_an(
        cfg=cfg, node_location=node_location, test_inputs=test_inputs, title=title, error_message=error_message
    )
    plotting_function(ax, pca_attn_outputs, title)

    major_tag = QType.MATH_ADD if operation == MathsToken.PLUS else QType.MATH_SUB # Does not handle NEG case
    cfg.add_useful_node_tag( node_location, major_tag.value, pca_op_tag(answer_digit, operation, True) )

def manual_nodes_pca(cfg, operation, nodes):
    print("Manual PCA tags for", cfg.model_name, "with operation", token_to_char(cfg, operation))

    cols = 4
    rows = 1 + (len(nodes)+1) // cols

    fig, axs = plt.subplots(rows, cols)
    fig.set_figheight(rows*2 + 1)
    fig.set_figwidth(10)

    index = 0
    for node in nodes:
        manual_node_pca(cfg=cfg, ax=axs[index // cols, index % cols], position=node[0],
                        layer=node[1], num=node[2], operation=operation, answer_digit=node[3],
                        plotting_function=plot_pca_for_an)
        index += 1

  # Remove any graphs we dont need (except last one)
    while index < rows * cols - 1:
        ax = axs[index // cols, index % cols]
        ax.remove()
        index += 1

    # Replace last graph with the legend
    lines_labels = [axs[0,0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    axs[rows-1, cols-1].legend(lines, labels)
    axs[rows-1, cols-1].axis('off') # Now, to hide the last subplot

    plt.tight_layout()
    save_plt_to_file(cfg=cfg, full_title='Pca Tr')
    plt.show()

def auto_node_pca(cfg, ax, index, node_location, operation, answer_digit, perc_threshold):

  pca, pca_attn_outputs, title = calc_pca_for_an(node_location, operation, answer_digit)
  if pca is not None:
    perc = pca_evr_0_percent(pca)
    if perc > perc_threshold:
      plot_pca_for_an(ax, pca_attn_outputs, title)

      major_tag = QType.MATH_ADD if operation == MathsToken.PLUS else QType.MATH_SUB # Does not handle NEG case
      cfg.add_useful_node_tag(node_location, major_tag.value, pca_op_tag(answer_digit, operation, False))
      return True

  return False