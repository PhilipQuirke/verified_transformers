import torch
from sklearn.decomposition import PCA
def pca_evr_0_percent(pca):
    return int(round(pca.explained_variance_ratio_[0] * 100, 0))


# Calculate one Principal Component Analysis
def calc_pca_for_an(cfg, node_location, test_inputs, title, error_message):
  assert node_location.is_head is True

  try:
    _, the_cache = cfg.main_model.run_with_cache(test_inputs)

    # Gather attention patterns for all the (randomly chosen) questions
    attention_outputs = []
    for i in range(len(test_inputs)):

      # Output of individual heads, without final bias
      attention_cache=the_cache["result", node_location.layer, "attn"] # Output of individual heads, without final bias
      attention_output=attention_cache[i]  # Shape [n_ctx, n_head, d_model]
      attention_outputs.append(attention_output[node_location.position, node_location.num, :])

    attn_outputs = torch.stack(attention_outputs, dim=0).cpu()

    pca = PCA(n_components=6)
    pca.fit(attn_outputs)
    pca_attn_outputs = pca.transform(attn_outputs)

    full_title = title + ', EVR[0]=' + str(pca_evr_0_percent((pca))) + '%'
    return pca, pca_attn_outputs, full_title
  except Exception as e:
      print(error_message, e)
      return None, None, None