import math
import torch
import numpy as np
import transformer_lens.utils as utils
import plotly.express as px
import plotly.graph_objects as go


# Plot multiple graph with lines
def plot_loss_lines(cfg, epochs_to_graph : int, raw_lines_list, x=None, mode='lines', labels=None, xaxis='Epoch', yaxis='Loss', title = '', log_y=False, hover=None, all_epochs=True, **kwargs):

    lines_list = raw_lines_list if all_epochs==False else [row[:epochs_to_graph] for row in raw_lines_list]
    log_suffix = '' if log_y==False else ' (Log)'
    epoch_suffix = '' if all_epochs==False else ' (' + str(epochs_to_graph) + ' Epochs)'
    full_title = title + log_suffix + epoch_suffix

    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    if cfg.save_graph_to_file :
      fig = go.Figure(layout={})
      print(full_title)
    else:
      fig = go.Figure(layout={'title':full_title})

    fig.update_xaxes(
        title=xaxis,
        showgrid=False)
    fig.update_yaxes(
        title=yaxis + log_suffix,
        showgrid=False)

    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = utils.to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))

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

      # Update x-axis ticks
      x_ticks = x[0::100]  # Start from index 0 and pick every 100th element
      x_ticks = x_ticks[1:] # Exclude the first tick (0)
      fig.update_xaxes(
          tickmode='array',
          tickvals=x_ticks,
          ticktext=[str(tick) for tick in x_ticks]
      )

    if cfg.save_graph_to_file:
        # fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),width=1200,height=300)
        # Update layout for legend positioning inside the graph
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            width=1200,height=300,
            legend=dict(
                x=0.92,  # Adjust this value to move the legend left or right
                y=0.99,  # Adjust this value to move the legend up or down
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
                bgcolor="White",  # Adjust background color for visibility
                bordercolor="Black",
                borderwidth=2
            ))

    fig.show(bbox_inches="tight")

    return full_title, fig