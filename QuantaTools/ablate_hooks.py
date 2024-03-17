import torch
import transformer_lens.utils as utils

from .ablate_config import AblateConfig, acfg
from .model_config import ModelConfig
from .model_loss import logits_to_tokens_loss, loss_fn
from .useful_node import NodeLocation, UsefulNode, UsefulNodeList


def a_null_attn_z_hook(value, hook):
  pass


def validate_value(name, value):
  if value.shape[0] == 0:
    print( "Aborted", name, acfg.node_names(), acfg.questions, acfg.operation, acfg.expected_answer, acfg.expected_impact)
    acfg.abort = True # TransformerLens returned a [0, 22, 3, 170] tensor. This is bad data. Bug in code? Abort
    return False

  return True


def a_get_l0_attn_z_hook(value, hook):
  # print( "In a_get_l0_attn_z_hook", value.shape) # Get [1, 22, 3, 170] = ???, cfg.n_ctx, cfg.n_heads, cfg.d_head
  if validate_value("a_get_l0_attn_z_hook", value):
    acfg.layer_store[0] = value.clone()

def a_get_l1_attn_z_hook(value, hook):
  if validate_value("a_get_l1_attn_z_hook", value):
    acfg.layer_store[1] = value.clone()

def a_get_l2_attn_z_hook(value, hook):
  if validate_value("a_get_l2_attn_z_hook", value):
    acfg.layer_store[2] = value.clone()

def a_get_l3_attn_z_hook(value, hook):
  if validate_value("a_get_l3_attn_z_hook", value):
    acfg.layer_store[3] = value.clone()


def a_put_l0_attn_z_hook(value, hook):
  # print( "In a_put_l0_attn_z_hook", value.shape) # Get [1, 22, 3, 170] = ???, cfg.n_ctx, cfg.n_heads, d_head
  for location in acfg.node_locations:
    if location.layer == 0:
      value[:,location.position,location.num,:] = acfg.layer_store[0][:,location.position,location.num,:].clone()

def a_put_l1_attn_z_hook(value, hook):
  for location in acfg.node_locations:
    if location.layer == 1:
      value[:,location.position,location.num,:] = acfg.layer_store[0][:,location.position,location.num,:].clone()

def a_put_l2_attn_z_hook(value, hook):
  for location in acfg.node_locations:
    if location.layer == 2:
      value[:,location.position,location.num,:] = acfg.layer_store[0][:,location.position,location.num,:].clone()

def a_put_l3_attn_z_hook(value, hook):
  for location in acfg.node_locations:
    if location.layer == 3:
      value[:,location.position,location.num,:] = acfg.layer_store[0][:,location.position,location.num,:].clone()


def a_reset(node_locations):
  acfg.reset_hooks()
  acfg.node_locations = node_locations
  acfg.attn_get_hooks = [(acfg.l_attn_hook_z_name[0], a_get_l0_attn_z_hook), (acfg.l_attn_hook_z_name[1], a_get_l1_attn_z_hook), (acfg.l_attn_hook_z_name[2], a_get_l2_attn_z_hook), (acfg.l_attn_hook_z_name[3], a_get_l3_attn_z_hook)][:cfg.n_layers]
  acfg.attn_put_hooks = [(acfg.l_attn_hook_z_name[0], a_put_l0_attn_z_hook), (acfg.l_attn_hook_z_name[1], a_put_l1_attn_z_hook), (acfg.l_attn_hook_z_name[2], a_put_l2_attn_z_hook), (acfg.l_attn_hook_z_name[3], a_put_l3_attn_z_hook)][:cfg.n_layers]


# Using the provided questions, run some model predictions and store the results in the cache, for use in later ablation interventions
def a_calc_mean_values(cfg, the_questions):

  # Run the sample batch, gather the cache
  cfg.main_model.reset_hooks()
  cfg.main_model.set_use_attn_result(True)
  sample_logits, sample_cache = cfg.main_model.run_with_cache(the_questions.cuda())
  print(sample_cache) # Gives names of datasets in the cache
  sample_losses_raw, sample_max_prob_tokens = logits_to_tokens_loss(cfg, sample_logits, the_questions.cuda())
  sample_loss_mean = utils.to_numpy(loss_fn(sample_losses_raw).mean())
  print("Sample Mean Loss", sample_loss_mean) # Loss < 0.04 is good


  # attn.hook_z is the "attention head output" hook point name (at a specified layer)
  sample_attn_z_0 = sample_cache[acfg.l_attn_hook_z_name[0]]
  print("Sample", acfg.l_attn_hook_z_name[0], sample_attn_z_0.shape) # gives [350, 22, 3, 170] = num_questions, cfg.n_ctx, n_heads, d_head
  acfg.mean_attn_z = torch.mean(sample_attn_z_0, dim=0, keepdim=True)
  print("Mean", acfg.l_attn_hook_z_name[0], acfg.mean_attn_z.shape) # gives [1, 22, 3, 170] = 1, cfg.n_ctx, n_heads, d_head


  # hook_resid_post is the "post residual memory update" hook point name (at a specified layer)
  sample_resid_post_0 = sample_cache[acfg.l_hook_resid_post_name[0]]
  print("Sample", acfg.l_hook_resid_post_name[0], sample_resid_post_0.shape) # gives [350, 22, 510] = num_questions, cfg.n_ctx, d_model
  acfg.mean_resid_post = torch.mean(sample_resid_post_0, dim=0, keepdim=True)
  print("Mean", acfg.l_hook_resid_post_name[0], acfg.mean_resid_post.shape) # gives [1, 22, 510] = 1, cfg.n_ctx, d_model


  # mlp.hook_post is the "MLP layer" hook point name (at a specified layer)
  sample_mlp_hook_post_0 = sample_cache[acfg.l_mlp_hook_post_name[0]]
  print("Sample", acfg.l_mlp_hook_post_name[0], sample_mlp_hook_post_0.shape) # gives [350, 22, 2040] = num_questions, cfg.n_ctx, cfg.d_mlp
  acfg.mean_mlp_hook_post = torch.mean(sample_mlp_hook_post_0, dim=0, keepdim=True)
  print("Mean", acfg.l_mlp_hook_post_name[0], acfg.mean_mlp_hook_post.shape) # gives [1, 22, 2040] = 1, cfg.n_ctx, cfg.d_mlp
