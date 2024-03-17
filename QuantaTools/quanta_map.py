import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import textwrap

from .useful_node import position_name, position_name_to_int, row_location_name, location_name, NodeLocation, UsefulNode, UsefulNodeList 


# Results to display in a quanta cell
class QuantaResult(NodeLocation):
    cell_text : str
    color_index : int

  
    def __init__(self, node, cell_text = "", color_index = 0):
        super().__init__(node.position, node.layer, node.is_head, node.num)  # Call the parent class's constructor    
        self.cell_text = cell_text    
        self.color_index = color_index


# Calculate the results to display in all the quanta cell
def calc_quanta_results( cfg, test_nodes : UsefulNodeList, major_tag : str, minor_tag : str, get_node_details, shades ):

    quanta_results = []
    
    for node in test_nodes.nodes:
        cell_text, color_index = get_node_details(cfg, node, major_tag, minor_tag, shades)
        if cell_text != "" :
            quanta_results +=[QuantaResult(node, cell_text, color_index)]

    return quanta_results


# Define a colormap for use with graphing
def create_custom_colormap():
    colors = ["green", "yellow"]
    return mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)


# Blend the color with white to make it paler
def pale_color(color, factor=0.5):
    color_array = np.array(color)
    white = np.array([1, 1, 1, 1])
    return white * factor + color_array * (1 - factor)


# Find the quanta result for the specified cell
def find_quanta_result_by_row_col(the_row_name, the_position, quanta_results):
    for result in quanta_results:
        if result.row_name() == the_row_name and result.position == the_position:
            return result
    return None
  

# Draw a cell in the specified color
def show_quanta_add_patch(ax, j, row, cell_color):
    ax.add_patch(plt.Rectangle((j, row), 1, 1, fill=True, color=cell_color))


# Calculate (but do not draw) the quanta map with cell contents provided by get_node_details 
def calc_quanta_map( cfg, standard_quanta : bool, shades, the_nodes : UsefulNodeList, major_tag : str, minor_tag : str, get_node_details, base_fontsize = 10, max_width = 10, left_reserve = 0.3 ):
  
    quanta_results = calc_quanta_results(cfg, the_nodes, major_tag, minor_tag, get_node_details, shades)

    distinct_row_names = set()
    distinct_positions = set()

    for result in quanta_results:
        distinct_row_names.add(result.row_name())
        distinct_positions.add(result.position)

    distinct_row_names = sorted(distinct_row_names)
    distinct_positions = sorted(distinct_positions)

    # Show standard_quanta (common across all potentional models) in blue shades and model-specific quanta in green shades 
    custom_cmap = plt.cm.winter if standard_quanta else create_custom_colormap()
  
    # Create figure and axes
    fig1, ax1 = plt.subplots(figsize=(2*len(distinct_positions)/3, 2*len(distinct_row_names)/3))  # Adjust the figure size as needed

    # Ensure cells are square
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_tick_params(labelleft=True, labelright=False)

    colors = [pale_color(custom_cmap(i/shades)) for i in range(shades)]
    horizontal_labels = []
    wrapper = textwrap.TextWrapper(width=max_width)


    show_row = len(distinct_row_names)-1
    for the_row_name in distinct_row_names:
        show_col = 0
        for the_position in distinct_positions:
            cell_color = 'lightgrey'  # Color for empty cells

            if show_row == 0:
                horizontal_labels += [cfg.token_position_meanings[the_position]]

            result = find_quanta_result_by_row_col(the_row_name, the_position, quanta_results)
            if result != None:
                cell_color = colors[result.color_index] if result.color_index >= 0 else 'lightgrey'
                the_fontsize = base_fontsize if len(result.cell_text) < 4 else base_fontsize-1 if len(result.cell_text) < 5 else base_fontsize-2
                wrapped_text = wrapper.fill(text=result.cell_text)
                ax1.text(show_col + 0.5, show_row + 0.5, wrapped_text, ha='center', va='center', color='black', fontsize=the_fontsize)

            show_quanta_add_patch(ax1, show_col, show_row, cell_color)
            show_col += 1

        show_row -= 1


    # Configure x axis
    ax1.set_xlim(0, len(horizontal_labels))
    ax1.set_xticks(np.arange(0.5, len(horizontal_labels), 1))
    ax1.set_xticklabels(horizontal_labels)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(axis='x', length=0)
    for label in ax1.get_xticklabels():
        label.set_fontsize(9)

    # Configure y axis
    distinct_row_names = distinct_row_names[::-1] # Reverse the order
    ax1.set_ylim(0, len(distinct_row_names))
    ax1.set_yticks(np.arange(0.5, len(distinct_row_names), 1))
    ax1.set_yticklabels(distinct_row_names)
    ax1.tick_params(axis='y', length=0)
    for label in ax1.get_yticklabels():
        label.set_horizontalalignment('left')
        label.set_position((-0.1, 0))  # Adjust the horizontal position

    # Reserve constant space for y-axis labels
    # fig1.subplots_adjust(left=left_reserve) Doesnt work as desired

  
    return ax1, quanta_results
