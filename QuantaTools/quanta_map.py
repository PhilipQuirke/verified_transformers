import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import textwrap

from .useful_node import position_name, NodeLocation, UsefulNodeList 


# Results to display in a quanta cell
class QuantaResult(NodeLocation):

  
    def __init__(self, node, cell_text = "", color_index = 0):
        super().__init__(node.position, node.layer, node.is_head, node.num)  # Call the parent class's constructor    
        self.cell_text : str = cell_text    
        self.color_index : int = color_index


# Calculate the results to display in all the quanta cell
def calc_quanta_results( cfg, test_nodes : UsefulNodeList, major_tag : str, minor_tag : str, get_node_details, num_shades : int ):

    quanta_results = []
    
    for node in test_nodes.nodes:
        cell_text, color_index = get_node_details(cfg, node, major_tag, minor_tag, num_shades)
        if cell_text != "" :
            quanta_results +=[QuantaResult(node, cell_text, color_index)]

    return quanta_results


# Show standard_quanta (common across all potential models) in blue shades and model-specific quanta in green/yellow num_shades 
def create_colormap(standard_quanta : bool):
    if standard_quanta:
        return plt.cm.winter
    
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
  

# Draw a cell background in the specified color
def show_quanta_patch(ax, col_idx, row_idx, cell_color):
    ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=True, color=cell_color))


# Draw the cell text
def show_quanta_text(ax, col_idx, row_idx, cell_text, the_fontsize):
    if cell_text != "":
        ax.text(col_idx + 0.5, row_idx + 0.5, cell_text, ha='center', va='center', color='black', fontsize=the_fontsize)


# Draw the cell border
def show_quanta_border(ax, merge_start, col_idx, row_idx, cell_color):
    ax.add_patch(patches.Rectangle((col_idx, merge_start), 1, row_idx - merge_start, edgecolor='black', facecolor=cell_color, lw=1))


# Calculate (but do not draw) the quanta map with cell contents provided by get_node_details 
def calc_quanta_map( cfg, standard_quanta : bool, num_shades : int, the_nodes : UsefulNodeList, major_tag : str, minor_tag : str, get_node_details, base_fontsize : int = 10, max_width : int = 10, combine_identical_cells : bool = True ):
  
    quanta_results = calc_quanta_results(cfg, the_nodes, major_tag, minor_tag, get_node_details, num_shades)

    distinct_row_names = set()
    distinct_positions = set()

    for result in quanta_results:
        distinct_row_names.add(result.row_name())
        distinct_positions.add(result.position)

    distinct_row_names = sorted(distinct_row_names)
    distinct_positions = sorted(distinct_positions)

    if len(distinct_row_names) == 0 or len(distinct_positions) == 0:
        return None, quanta_results, 0


    # Show standard_quanta (common across all potentional models) in blue num_shades and model-specific quanta in green num_shades 
    colormap = create_colormap(standard_quanta)
  
    # Create figure and axes
    _, ax1 = plt.subplots(figsize=(2*len(distinct_positions)/3, 2*len(distinct_row_names)/3))  # Adjust the figure size as needed

    # Ensure cells are square
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_tick_params(labelleft=True, labelright=False)

    colors = [pale_color(colormap(i/num_shades)) for i in range(num_shades)]
    horizontal_top_labels = []
    horizontal_bottom_labels = []
    wrapper = textwrap.TextWrapper(width=max_width)

    num_results = 0
    col_idx = 0
    for the_position in distinct_positions:
        previous_text = None
        merge_start = None
        cell_color = 'lightgrey'
        
        row_idx = len(distinct_row_names)-1
        for the_row_name in distinct_row_names:

            if row_idx == 0:
                horizontal_top_labels += [cfg.token_position_meanings[the_position]]
                horizontal_bottom_labels += [position_name(the_position)]

            result = find_quanta_result_by_row_col(the_row_name, the_position, quanta_results)
            if result != None:
                num_results += 1
                the_shade = max(0, min(result.color_index, num_shades-1))
                cell_color = colors[the_shade] if result.color_index >= 0 else 'lightgrey'
                the_fontsize = base_fontsize if len(result.cell_text) < 4 else base_fontsize-1 if len(result.cell_text) < 5 else base_fontsize-2
                cell_text = wrapper.fill(text=result.cell_text)
                #show_quanta_text( ax1, col_idx + 0.5, row_idx + 0.5, cell_text, the_fontsize)
            else:
                cell_text = ""
                cell_color = 'lightgrey'  # Color for empty cells

            #show_quanta_patch(ax1, col_idx, row_idx, cell_color)          
        
            # Check if current cell text matches the previous cell text
            if combine_identical_cells and cell_text == previous_text and row_idx != len(distinct_row_names) - 1:
                continue

            # Draw the previous sequence of similar cells
            if previous_text and merge_start is not None:
                show_quanta_border(ax1, merge_start, col_idx, row_idx, cell_color)
                show_quanta_text( ax1, col_idx + 0.5, (merge_start + row_idx) / 2, previous_text, the_fontsize)
        
            # Update trackers
            merge_start = row_idx
            previous_text = cell_text
        
            row_idx -= 1
            

        # Draw the last sequence of similar cells
        if previous_text:
            show_quanta_border(ax1, merge_start, col_idx, row_idx, cell_color)
            #ax1.add_patch(patches.Rectangle((col_idx, merge_start), 1, len(distinct_row_names) - merge_start, edgecolor='black', facecolor=cell_color, lw=1))
            show_quanta_text( ax1, col_idx + 0.5, (merge_start + len(distinct_row_names)) / 2, previous_text, the_fontsize)

        col_idx += 1


    # Configure x axis
    ax1.set_xlim(0, len(horizontal_top_labels))
    ax1.set_xticks(np.arange(0.5, len(horizontal_top_labels), 1))
    ax1.set_xticklabels(horizontal_top_labels)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(axis='x', length=0)
    for label in ax1.get_xticklabels():
        label.set_fontsize(9)


    # Add the extra row of labels (P0, P5, P8, ...) below the matrix
    for index in range(len(horizontal_bottom_labels)):
        label = horizontal_bottom_labels[index]
        ax1.text(index + 0.5, - 0.02, label, ha='center', va='top', fontsize=9, transform=ax1.get_xaxis_transform())

    # Adjust figure layout to accommodate the new row of labels
    plt.subplots_adjust(bottom=0.1)  # Adjust as needed based on your specific figure layout


    # Configure y axis
    distinct_row_names = distinct_row_names[::-1] # Reverse the order
    ax1.set_ylim(0, len(distinct_row_names))
    ax1.set_yticks(np.arange(0.5, len(distinct_row_names), 1))
    ax1.set_yticklabels(distinct_row_names)
    ax1.tick_params(axis='y', length=0)
    for label in ax1.get_yticklabels():
        label.set_horizontalalignment('left')
        label.set_position((-0.1, 0))  # Adjust the horizontal position
  
    return ax1, quanta_results, num_results
