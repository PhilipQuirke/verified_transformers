from .useful_node import NodeLocation

from .ablate_config import acfg
from .ablate_hooks import a_predict_questions


def ablate_mlp_hook_post(value, hook):
  # print( "In ablate_mlp_hook_post", value.shape) # Get [1099, 22, 2040] = num_varied_questions, cfg.n_ctx, cfg.d_mlp (# neurons)
  position = acfg.ablate_node_locations[0].position

  # Mean ablate. Copy the mean resid post values in the MLP layer
  value[:,position,:] = acfg.mean_mlp_hook_post[:,position,:].clone()


# Ablate the MLP in each layer in each position. If the loss increases, the layer+MLP is useful to the model.
def ablate_mlp_and_add_useful_node_tags(cfg, questions, test_questions_and_add_useful_node_tags):
  for position in cfg.useful_positions:
    for layer in range(cfg.n_layers):
      for num in range(cfg.mlp_slices):

        node_location = NodeLocation(position, layer, False, num) # Ablate this node 
        acfg.ablate_node_locations = [node_location]  # Ablate this node  

        the_hooks = [(acfg.l_mlp_hook_post_name[layer], ablate_mlp_hook_post)]
        all_losses_raw, all_max_prob_tokens = a_predict_questions(cfg, questions, the_hooks)
    
        # Test accuracy of model in predicting question answers, when a single node is ablated.
        # Adds nodes to Useful.useful_nodes and adds tags to those nodes.
        test_questions_and_add_useful_node_tags(cfg, acfg, questions, node_location, all_losses_raw, all_max_prob_tokens)

        
def ablate_head_attn_hook_z(value, hook):
  # print( "In ablate_head_attn_hook_z", value.shape) # Get [1, 22, 3, 170] = ???, cfg.n_ctx, cfg.n_heads, cfg.d_head
  position = acfg.ablate_node_locations[0].position
  head = acfg.ablate_node_locations[0].num

  # Mean ablate. Copy the mean resid post values in position N to all the batch questions
  value[:,position,head,:] = acfg.mean_attn_z[:,position,head,:].clone()


# Ablate each head in each layer in each position. If the loss increases, the position+layer+head is useful to the algorithm.
def ablate_head_and_add_useful_node_tags(cfg, questions, test_questions_and_add_useful_node_tags):
  for position in cfg.useful_positions:
    for layer in range(cfg.n_layers):
      for attn_head in range(cfg.n_heads):

        node_location = NodeLocation(position, layer, True, attn_head) # Ablate this node 
        acfg.ablate_node_locations = [node_location]
        
        the_hooks = [(acfg.l_attn_hook_z_name[layer], ablate_head_attn_hook_z)]
        all_losses_raw, all_max_prob_tokens = a_predict_questions(cfg, questions, the_hooks)
        
        # Test accuracy of model in predicting question answers, when a single node is ablated.
        # Adds nodes to Useful.useful_nodes and adds tags to those nodes.
        test_questions_and_add_useful_node_tags(cfg, acfg, questions, node_location, all_losses_raw, all_max_prob_tokens)