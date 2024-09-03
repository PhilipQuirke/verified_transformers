import torch
import transformer_lens.utils as utils

from .model_token_to_char import tokens_to_string
from .model_train import logits_to_tokens_loss, loss_fn
from .quanta_map_impact import get_answer_impact
from .quanta_constants import NO_IMPACT_TAG
from .ablate_config import acfg



# (Question and answer) position ablation.
# Impacts all nodes at acfg.ablate_node_locations[0].position
def a_put_resid_post_hook(value, hook):
    #print( "In l_hook_resid_post_name", value.shape, acfg.ablate_node_locations[0].position) # Get [64, 22, 510] = cfg.batch_size, cfg.n_ctx, d_model

    # Copy the mean resid post values in position N to all the layers
    value[:,acfg.ablate_node_locations[0].position,:] = acfg.mean_resid_post[0,acfg.ablate_node_locations[0].position,:].clone()
    

def validate_value(name, value):
    if value.shape[0] == 0:
        print( "Aborted", name, acfg.ablate_node_names, acfg.operation, acfg.expected_answer, acfg.expected_impact)
        acfg.abort = True # TransformerLens returned a [0, 22, 3, 170] tensor. This is bad data. Bug in code? Abort
        return False

    return True


def a_get_l0_attn_z_hook(value, hook):
    #print( "In a_get_l0_attn_z_hook", value.shape ) # Get [1, 22, 3, 170] = ???, cfg.n_ctx, cfg.n_heads, cfg.d_head
    if validate_value("a_get_l0_attn_z_hook", value):
        acfg.layer_store[0] = value.clone()

def a_get_l1_attn_z_hook(value, hook):
    #print( "In a_get_l1_attn_z_hook", value.shape ) # Get [1, 22, 3, 170] = ???, cfg.n_ctx, cfg.n_heads, cfg.d_head
    if validate_value("a_get_l1_attn_z_hook", value):
        acfg.layer_store[1] = value.clone()

def a_get_l2_attn_z_hook(value, hook):
    if validate_value("a_get_l2_attn_z_hook", value):
        acfg.layer_store[2] = value.clone()

def a_get_l3_attn_z_hook(value, hook):
    if validate_value("a_get_l3_attn_z_hook", value):
        acfg.layer_store[3] = value.clone()


def a_put_l0_attn_z_hook(value, hook):
    for location in acfg.ablate_node_locations:
        if location.is_head and location.layer == 0:
            #print( "In a_put_l0_attn_z_hook", value.shape, location.name()) # Get [1, 22, 3, 170] = ???, cfg.n_ctx, cfg.n_heads, d_head
            value[:,location.position,location.num,:] = acfg.layer_store[0][:,location.position,location.num,:].clone()

def a_put_l1_attn_z_hook(value, hook):
    for location in acfg.ablate_node_locations:
        if location.is_head and location.layer == 1:
            #print( "In a_put_l1_attn_z_hook", value.shape, location.name()) # Get [1, 22, 3, 170] = ???, cfg.n_ctx, cfg.n_heads, d_head
            value[:,location.position,location.num,:] = acfg.layer_store[1][:,location.position,location.num,:].clone()

def a_put_l2_attn_z_hook(value, hook):
    for location in acfg.ablate_node_locations:
        if location.is_head and location.layer == 2:
            value[:,location.position,location.num,:] = acfg.layer_store[2][:,location.position,location.num,:].clone()

def a_put_l3_attn_z_hook(value, hook):
    for location in acfg.ablate_node_locations:
        if location.is_head and location.layer == 3:
            value[:,location.position,location.num,:] = acfg.layer_store[3][:,location.position,location.num,:].clone()
  

# Set acfg ablation hooks. Updates global acfg variable
def a_set_ablate_hooks(cfg):
    acfg.resid_put_hooks = [(acfg.l_hook_resid_post_name[0], a_put_resid_post_hook),(acfg.l_hook_resid_post_name[1], a_put_resid_post_hook),(acfg.l_hook_resid_post_name[2], a_put_resid_post_hook),(acfg.l_hook_resid_post_name[3], a_put_resid_post_hook)][:cfg.n_layers]

    acfg.attn_get_hooks = [(acfg.l_attn_hook_z_name[0], a_get_l0_attn_z_hook), (acfg.l_attn_hook_z_name[1], a_get_l1_attn_z_hook), (acfg.l_attn_hook_z_name[2], a_get_l2_attn_z_hook), (acfg.l_attn_hook_z_name[3], a_get_l3_attn_z_hook)][:cfg.n_layers]
    acfg.attn_put_hooks = [(acfg.l_attn_hook_z_name[0], a_put_l0_attn_z_hook), (acfg.l_attn_hook_z_name[1], a_put_l1_attn_z_hook), (acfg.l_attn_hook_z_name[2], a_put_l2_attn_z_hook), (acfg.l_attn_hook_z_name[3], a_put_l3_attn_z_hook)][:cfg.n_layers]


# Using the provided questions, run some model predictions and store the results in the cache, for use in later ablation interventions
def a_calc_mean_values(cfg, the_questions):

    # Run the sample batch, gather the cache
    cfg.main_model.reset_hooks()
    cfg.main_model.set_use_attn_result(True)
    sample_logits, sample_cache = cfg.main_model.run_with_cache(the_questions.cuda())
    print("Cache names", sample_cache) # Gives names of datasets in the cache
    sample_losses_raw, _ = logits_to_tokens_loss(cfg, sample_logits, the_questions.cuda())
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
 


# Ask the model to predict the question answers (with the hooks either getting data, putting intervention ablations in place, or doing nothing )
def a_predict_questions(cfg, questions, the_hooks):

    cfg.main_model.reset_hooks()
    cfg.main_model.set_use_attn_result(True)

    all_logits = None
    if the_hooks == None:
        all_logits, _ = cfg.main_model.run_with_cache(questions.cuda())
    else:
        all_logits = cfg.main_model.run_with_hooks(questions.cuda(), return_type="logits", fwd_hooks=the_hooks)
    all_losses_raw, all_max_prob_tokens = logits_to_tokens_loss(cfg, all_logits, questions.cuda())

    return all_losses_raw, all_max_prob_tokens 


# Run an ablation intervention on the model, and return a description of the impact of the intervention
# In the https://arxiv.org/pdf/2404.15255 terminology, this is a "noising" ablation.
def a_run_attention_intervention(cfg, store_question_and_answer, clean_question_and_answer, clean_answer_str):

    # These are all matrixes of tokens
    store_question = store_question_and_answer[:cfg.num_question_positions]
    clean_question = clean_question_and_answer[:cfg.num_question_positions]
    
    acfg.num_tests_run += 1

    description = "CleanAns: " + clean_answer_str + ", ExpectedAns/Impact: " + acfg.expected_answer + "/" + acfg.expected_impact + ", "


    a_predict_questions(cfg, store_question, acfg.attn_get_hooks)
    if acfg.abort:
        return description + "(Aborted on store)"


    # Predict "test" question overriding PnLmHp to give a bad answer
    all_losses_raw, all_max_prob_tokens = a_predict_questions(cfg, clean_question, acfg.attn_put_hooks)
    if acfg.abort:
        return description + "(Aborted on intervention)"
    if all_losses_raw.shape[0] == 0:
        acfg.abort = True
        print( "Bad all_losses_raw", all_losses_raw.shape, store_question_and_answer, clean_question_and_answer )
        return description + "(Aborted on Bad all_losses_raw)"
    loss_max = utils.to_numpy(loss_fn(all_losses_raw[0]).max())
    acfg.intervened_answer = tokens_to_string(cfg, all_max_prob_tokens[0])


    # Compare the clean test question answer to what the model generated (impacted by the ablation intervention)
    assert len(clean_answer_str) == len(acfg.intervened_answer)
    acfg.intervened_impact = get_answer_impact( cfg, clean_answer_str, acfg.intervened_answer )
    if acfg.intervened_impact == "":
        acfg.intervened_impact = NO_IMPACT_TAG

    description += "AblatedAns/Impact: " + acfg.intervened_answer + "/" + acfg.intervened_impact

    if loss_max > acfg.threshold:
        loss_str = NO_IMPACT_TAG if loss_max < 1e-7 else str(loss_max)
        description += ", Loss: " + loss_str

    return description
