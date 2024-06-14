import math
import torch
import numpy as np
import transformer_lens.utils as utils
import plotly.express as px
import plotly.graph_objects as go


def plot_loss_lines_layout(cfg, fig, tick_font_size, x, width=1200, height=400):
  
    num_points = len(x)
    tick_interval = 100 if num_points <= 1501 else 1000 if num_points <= 10001 else 5000 

    # Update x-axis ticks
    x_ticks = x[0::tick_interval]  # Start from index 0 and pick every 100th element
    x_ticks = x_ticks[1:] # Exclude the first tick (0)
    fig.update_xaxes(
        tickmode='array',
        tickvals=x_ticks,
        ticktext=[str(tick) for tick in x_ticks],
        tickfont=dict(size=tick_font_size)  # Set the font size for x-axis ticks        
    )
      
    # Update layout for legend positioning inside the graph
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        width=width,height=height,
        legend=dict(
            x=0.92,  # Adjust this value to move the legend left or right
            y=0.99,  # Adjust this value to move the legend up or down
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=tick_font_size,
                color="black"
            ),
            bgcolor="White",  # Adjust background color for visibility
            bordercolor="Black",
            borderwidth=2
        ))

    fig.show(bbox_inches="tight")


# Plot multiple graph with lines
def plot_loss_lines(cfg, steps_to_graph : int, raw_lines_list, 
                    x=None, mode='lines', labels=None, xaxis='Training Steps', 
                    yaxis='Loss', title = '', log_y=False, 
                    all_steps=True, title_font_size=20, tick_font_size=12, 
                    width=1200, height=400,
                    **kwargs):

    lines_list = raw_lines_list if all_steps==False else [row[:steps_to_graph] for row in raw_lines_list]
    the_prefix = '' if log_y==False else 'Log '
    the_suffix = '' if all_steps==False else ' (' + str(steps_to_graph) + ' training steps)'
    full_title = the_prefix + title + the_suffix

    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    if cfg.graph_file_suffix != "" :
      fig = go.Figure(layout={})
      print(full_title)
    else:
      fig = go.Figure(layout={'title':full_title})

    fig.update_xaxes(
        title=xaxis,
        title_font={"size": title_font_size},   
        tickfont=dict(size=tick_font_size),       
        showgrid=False)
    fig.update_yaxes(
        title=the_prefix + yaxis,
        title_font={"size": title_font_size},       
        tickfont=dict(size=tick_font_size),       
        showgrid=False)

    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = utils.to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label))

    if log_y:
        fig.update_layout(yaxis_type="log")
    else:
      # Calculate the max y-value rounded up to the nearest integer
      y_max = 1
      for k in range(len(lines_list)):
          y_max = max(y_max, math.ceil(max(lines_list[k])) )
      y_max = 3 # manual override if necessary

      # Update layout to set the y-axis min to 0 and max to the calculated y_max
      fig.update_layout(
          yaxis=dict(range=[0, y_max])
      )

    plot_loss_lines_layout(cfg, fig, tick_font_size=tick_font_size, x=x, width=width,height=height)

    return full_title, fig