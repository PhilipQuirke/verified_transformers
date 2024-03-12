class QuantaType:
  # Token position tag
  POSITION = "Position"
  
  # What % of questions failed when we ablated a specific node. A low percentage indicates a less common use case
  FAIL = "Fail%"
  
  # What input tokens (e.g. "D'3") are attended to by a specific attention head.
  ATTENTION = "Attn"
  
  # What answer digits (e.g. "A543") were impacted when we ablated a specific node.
  IMPACT = "Impact"
  
  # What does Principal Component Analysis say about the node?
  PCA = "PCA"
  
  # What algorithmic purpose is this node serving?
  ALGO = "Algorithm"


# For each node, we store at most 5 input attention facts (as tags)
MAX_ATTENTION_TAGS = 5

# When graphing, we only show input tokens with > 10% of the node's attention
MIN_ATTENTION_PERC = 10
