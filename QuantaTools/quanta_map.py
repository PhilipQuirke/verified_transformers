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
  

# Draw the cell background in the specified color
def show_quanta_patch(ax, col : float, row : float, cell_color : str, width : int = 1, height : int = 1):
    ax.add_patch(plt.Rectangle((col, row), width, height, fill=True, color=cell_color))


# Draw one cell's text
def show_quanta_text(ax, col : float, row : float, text : str, base_fontsize : int):
    if (text) and (text != ""):
        the_fontsize = base_fontsize if len(text) < 4 else base_fontsize-1 if len(text) < 6 else base_fontsize-2      
        ax.text(col + 0.5, row + 1, text, ha='center', va='center', color='black', fontsize=the_fontsize)


# Draw a thin border around 2 to 8 cells in a vertical column
def show_quanta_border(ax, col : float, start_row : float, end_row : float):
    num_cells = end_row - start_row + 1
    if num_cells > 1:
        print( "Border around", num_cells, "cells at", col, start_row, end_row)
        ax.add_patch(patches.Rectangle((col, start_row+1), 1, end_row - start_row, edgecolor='black', fill=False, lw=1))


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

    num_rows = len(distinct_row_names)     
    num_cols = len(distinct_positions)     
    if num_rows == 0 or num_cols == 0:
        return None, quanta_results, 0


    # Show standard_quanta (common across all potentional models) in blue num_shades and model-specific quanta in green num_shades 
    colormap = create_colormap(standard_quanta)
  
    # Create figure and axes
    _, ax1 = plt.subplots(figsize=(2*num_cols/3, 2*num_rows/3))  # Adjust the figure size as needed

    # Ensure cells are square
    ax1.set_aspect('equal', adjustable='box')
    ax1.yaxis.set_tick_params(labelleft=True, labelright=False)

    colors = [pale_color(colormap(i/num_shades)) for i in range(num_shades)]
    horizontal_top_labels = []
    horizontal_bottom_labels = []
    wrapper = textwrap.TextWrapper(width=max_width)

    num_results = 0
    
    show_quanta_patch(ax1, 0, 0, "lightgrey", num_cols, num_rows)  # Color for empty cells
    

    # Iterate over positions (columns)
    for col_idx, the_position in enumerate(distinct_positions):

        horizontal_top_labels += [cfg.token_position_meanings[the_position]]
        horizontal_bottom_labels += [position_name(the_position)]

        previous_text = None
        merge_start_row = None
        
        # Iterate over rows (layer + head or neuron) in reverse order      
        row_idx = num_rows - 1
        for the_row_name in distinct_row_names:
            cell_text = None

            result = find_quanta_result_by_row_col(the_row_name, the_position, quanta_results)
            if result != None:
                num_results += 1
                cell_text = wrapper.fill(text=result.cell_text)
                if result.color_index >= 0:
                    cell_color = colors[max(0, min(result.color_index, num_shades-1))] 
                    show_quanta_patch(ax1, col_idx, row_idx, cell_color)          
        
            # Check if current cell text matches the previous cell text
            if combine_identical_cells and (cell_text) and (cell_text == previous_text) and (row_idx != num_rows - 1):
                continue

            # Draw the previous sequence of similar cells
            if previous_text:
                show_quanta_border(ax1, col_idx, merge_start_row, row_idx)
                show_quanta_text( ax1, col_idx, (merge_start_row + row_idx) / 2.0, previous_text, base_fontsize)
                previous_text = None
                merge_start_row = None       
                
            # Update trackers
            previous_text = cell_text
            merge_start_row = row_idx
        
            row_idx -= 1
            

        # Draw the last sequence of similar cells
        if previous_text:
            show_quanta_border(ax1, col_idx, merge_start_row, row_idx)
            show_quanta_text( ax1, col_idx, (merge_start_row + num_rows) / 2, previous_text, base_fontsize)


    # Configure x axis
    ax1.set_xlim(0, num_cols)
    ax1.set_xticks(np.arange(0.5, num_cols, 1))
    ax1.set_xticklabels(horizontal_top_labels)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(axis='x', length=0)
    for label in ax1.get_xticklabels():
        label.set_fontsize(9)


    # Add the extra row of labels (P0, P5, P8, ...) below the matrix
    for index in range(num_cols):
        label = horizontal_bottom_labels[index]
        ax1.text(index + 0.5, - 0.02, label, ha='center', va='top', fontsize=9, transform=ax1.get_xaxis_transform())

    # Adjust figure layout to accommodate the new row of labels
    plt.subplots_adjust(bottom=0.1)  # Adjust as needed based on your specific figure layout


    # Configure y axis
    distinct_row_names = distinct_row_names[::-1] # Reverse the order
    ax1.set_ylim(0, num_rows)
    ax1.set_yticks(np.arange(0.5, num_rows, 1))
    ax1.set_yticklabels(distinct_row_names)
    ax1.tick_params(axis='y', length=0)
    for label in ax1.get_yticklabels():
        label.set_horizontalalignment('left')
        label.set_position((-0.1, 0))  # Adjust the horizontal position
  
    return ax1, quanta_results, num_results
