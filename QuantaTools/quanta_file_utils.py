import matplotlib.pyplot as plt

def save_plt_to_file(cfg, full_title):
  if cfg.graph_file_suffix > "":
    filename = cfg.file_config_prefix + full_title + '.' + cfg.graph_file_suffix
    filename = filename.replace(" ", "").replace(",", "").replace(":", "").replace("-", "")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)