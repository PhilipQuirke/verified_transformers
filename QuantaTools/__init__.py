# model_*.py: Contains the configuration of the transformer model being trained/analysed
from .model_config import ModelConfig
from .model_token_to_char import token_to_char, tokens_to_string
from .model_loss import logits_to_tokens_loss, loss_fn


# useful_*.py: Contains data on the useful token positions and useful nodes (attention heads and MLP neurons) that the model uses in predictions
from .useful_config import UsefulConfig 
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, answer_name, NodeLocation, str_to_node_location, UsefulNode, UsefulNodeList


# quanta_*.py: Contains categorisations of model behavior (aka quanta). Applicable to all models
from .quanta_constants import QCondition, QType, MAX_ATTN_TAGS, MIN_ATTN_PERC, NO_IMPACT_TAG, FAIL_SHADES, ATTN_SHADES, ALGO_SHADES, MATH_ADD_SHADES, MATH_SUB_SHADES
from .quanta_filter import FilterNode, FilterAnd, FilterOr, FilterHead, FilterNeuron, FilterContains, FilterPosition, FilterAttention, FilterImpact, FilterPCA, FilterAlgo, filter_nodes


# ablate_*.py: Contains ways to "intervention ablate" the model and detect the impact of the ablation
from .ablate_config import AblateConfig, acfg
from .ablate_hooks import a_put_resid_post_hook, a_set_ablate_hooks, a_calc_mean_values, a_predict_questions, a_run_attention_intervention
from .ablate_add_useful import ablate_mlp_and_add_useful_node_tags, ablate_head_and_add_useful_node_tags


# quanta_*.py: Contains ways to detect and graph model behavior (aka quanta) 
from .quanta_test_questions import test_questions_and_add_node_attention_tags
from .quanta_map import create_colormap, calc_quanta_map
from .quanta_map_attention import get_quanta_attention
from .quanta_map_failperc import get_quanta_fail_perc
from .quanta_map_binary import get_quanta_binary
from .quanta_map_impact import get_answer_impact, get_question_answer_impact, is_answer_sequential, compact_answer_if_sequential, get_quanta_impact, sort_unique_digits


# algo_*.py: Contains utilities to support model algorithm investigation
from .algo_config import AlgoConfig, search_and_tag_digit_position, search_and_tag_digit, search_and_tag


# maths_*.py: Contains specializations of the above specific to arithmetic (addition and subtraction) transformer models
from .maths_config import MathsConfig
from .maths_constants import MathsToken, MathsBehavior, MathsAlgorithm 
from .maths_utilities import set_maths_vocabulary, set_maths_question_meanings, int_to_answer_str, tokens_to_unsigned_int, tokens_to_answer, insert_question_number, make_a_maths_question_and_answer
from .maths_complexity import get_maths_question_complexity, get_maths_min_complexity, calc_maths_quanta_for_position_nodes
from .maths_data_generator import maths_data_generator_core, maths_data_generator, make_maths_questions_and_answers
from .maths_test_questions import make_maths_test_questions_and_answers, test_maths_questions_by_complexity, test_maths_questions_by_impact, test_maths_questions_and_add_useful_node_tags, TRICASE_QUESTIONS, make_maths_tricase_questions
