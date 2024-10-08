# model_*.py: Contains the configuration of the transformer model being trained/analysed
from .model_config import ModelConfig
from .model_token_to_char import token_to_char, tokens_to_string
from .model_train import logits_to_tokens_loss, loss_fn, get_training_optimizer_and_scheduler
from .model_train_json import download_huggingface_json, load_training_json
from .model_loss_graph import plot_loss_lines, plot_loss_lines_layout


from .model_sae import AdaptiveSparseAutoencoder, save_sae_to_huggingface 
from .model_sae_train import analyze_mlp_with_sae, optimize_sae_hyperparameters
from .model_sae_graph import analyze_and_visualize_sae


# useful_*.py: Contains data on the useful token positions and useful nodes (attention heads and MLP neurons) that the model uses in predictions
from .useful_config import UsefulConfig 
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, answer_name, NodeLocation, str_to_node_location, UsefulNode, UsefulNodeList


# quanta_*.py: Contains categorisations of model behavior (aka quanta). Applicable to all models
from .quanta_constants import QCondition, QType, MAX_ATTN_TAGS, MIN_ATTN_PERC, NO_IMPACT_TAG, FAIL_SHADES, ATTN_SHADES, ALGO_SHADES, MATH_ADD_SHADES, MATH_SUB_SHADES
from .quanta_file_utils import save_plt_to_file
from .quanta_filter import FilterNode, FilterAnd, FilterOr, FilterName, FilterTrue, FilterHead, FilterNeuron, FilterContains, FilterPosition, FilterLayer, FilterAttention, FilterImpact, FilterAlgo, filter_nodes, print_algo_purpose_results


# ablate_*.py: Contains ways to "intervention ablate" the model and detect the impact of the ablation
from .ablate_config import AblateConfig, acfg
from .ablate_hooks import a_put_resid_post_hook, a_set_ablate_hooks, a_calc_mean_values, a_predict_questions, a_run_attention_intervention
from .ablate_add_useful import ablate_mlp_and_add_useful_node_tags, ablate_head_and_add_useful_node_tags

# model_pca.py: Ways to extract PCA information from model
from .model_pca import calc_pca_for_an, pca_evr_0_percent


# quanta_*.py: Contains ways to detect and graph model behavior (aka quanta) 
from .quanta_add_attn_tags import add_node_attention_tags
from .quanta_map import create_colormap, calc_quanta_map
from .quanta_map_attention import get_quanta_attention
from .quanta_map_failperc import get_quanta_fail_perc
from .quanta_map_binary import get_quanta_binary, get_quanta_algo
from .quanta_map_impact import get_answer_impact, get_question_answer_impact, is_answer_sequential, compact_answer_if_sequential, get_quanta_impact, sort_unique_digits


# algo_*.py: Contains utilities to support model algorithm investigation
from .algo_config import AlgoConfig
from .algo_search import search_and_tag_digit_position, search_and_tag_digit, search_and_tag


# maths_*.py: Contains specializations of the above specific to arithmetic (addition and subtraction) transformer models
from .maths_tools.maths_config import MathsConfig
from .maths_tools.maths_constants import MathsToken, MathsBehavior, MathsTask
from .maths_tools.maths_utilities import (set_maths_vocabulary, set_maths_question_meanings, int_to_answer_str, 
    tokens_to_unsigned_int, tokens_to_answer, insert_question_number, make_a_maths_question_and_answer)
from .maths_tools.maths_complexity import (get_maths_question_complexity, get_maths_min_complexity, 
    calc_maths_quanta_for_position_nodes, get_maths_operation_complexity, get_maths_nodes_operation_coverage)
from .maths_tools.maths_data_generator import (maths_data_generator_addition, maths_data_generator_subtraction, 
    maths_data_generator_multiplication, maths_data_generator_mixed, maths_data_generator_mixed_core, 
    maths_data_generator, make_maths_questions_and_answers, MixedMathsDataset, get_mixed_maths_dataloader)
from .maths_tools.maths_test_questions import make_maths_test_questions_and_answers
from .maths_tools.maths_test_questions.test_questions_checker import (test_maths_questions_by_complexity, 
    test_maths_questions_by_impact, test_maths_questions_and_add_useful_node_tags, test_correctness_on_num_questions )
from .maths_tools import make_maths_tricase_questions, make_maths_tricase_questions_customized
from .maths_tools.maths_search_mix import (SubTaskBaseMath,
    run_intervention_core, run_strong_intervention, run_weak_intervention, 
    opr_functions, sgn_functions )
from .maths_tools.maths_search_add import (
    add_ss_functions, add_sc_functions, add_sa_functions, add_st_functions )
from .maths_tools.maths_search_sub import (
    sub_mt_functions, sub_gt_functions, sub_md_functions, sub_mb_functions, neg_nd_functions, neg_nb_functions )
from .maths_tools.maths_pca import (
    manual_nodes_pca, manual_node_pca, plot_nodes_pca_start, plot_nodes_pca_start_core, plot_nodes_pca_end )

