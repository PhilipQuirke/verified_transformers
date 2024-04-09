import random
import torch
import transformer_lens.utils as utils
from tqdm.notebook import tqdm

from .model_loss import logits_to_tokens_loss, loss_fn
from .model_token_to_char import tokens_to_string

from .useful_node import NodeLocation

from .quanta_constants import QType
from .quanta_map_impact import get_question_answer_impact, sort_unique_digits

from .ablate_config import AblateConfig, acfg
from .ablate_hooks import a_predict_questions

from .maths_constants import MathsToken, MathsBehavior
from .maths_data_generator import maths_data_generator, maths_data_generator_core, make_maths_questions_and_answers
from .maths_complexity import get_maths_question_complexity


# Move the data to the GPU (if any) for faster processing
def to_cuda(cfg, data):
    if cfg.use_cuda:
        return data.cuda()
    return data    


# Make addition BaseAdd questions
def make_maths_s0_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg, 
        MathsToken.PLUS, QType.MATH_ADD, MathsBehavior.ADD_S0_TAG,
        [[0, 0],
        [1, 3],
        [12345, 33333],
        [33333, 12345],
        [45762, 33113],
        [888, 11111],
        [2362, 23123],
        [15, 81],
        [1000, 4441],
        [4440, 11111],
        [24033, 25133],
        [23533, 21133],
        [32500, 1],
        [31500, 1111],
        [5500, 12323],
        [4500, 2209],
        [33345, 66643], # =099988
        [66643, 33345], # =099988
        [10770, 44111],
        [60000, 31111],
        [10000, 21111],
        [107700, 441111],
        [600000, 311111],
        [100000, 211111],
        [1077000, 4411111],
        [6000000, 3111111],
        [1000000, 2111111],
        [10770000, 44111111],
        [60000000, 3111111],
        [10000000, 2111111]])


# Make addition UseCarry1 questions
def make_maths_s1_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg, 
            MathsToken.PLUS, QType.MATH_ADD, MathsBehavior.ADD_S1_TAG,
            [[ 15, 45],
            [ 27, 55],
            [ 35, 59],
            [ 150, 451],
            [ 270, 551],
            [ 350, 591],
            [ 1500, 4511],
            [ 2700, 5511],
            [ 3500, 5911],
            [ 40035, 41149],
            [ 44000, 46000],
            [ 70000, 41111],
            [ 15000, 25111],
            [ 35000, 35111],
            [ 45000, 35111],
            [ 67000, 25111],
            [ 19000, 76111],
            [ 15020, 45091],
            [ 25002, 55019],
            [ 35002, 59019],
            [ 150211, 450911],
            [ 250021, 550191],
            [ 350021, 590191],
            [ 1502111, 4509111],
            [ 2500211, 5501911],
            [ 3500211, 5901911],
            [ 15021111, 45091111],
            [ 25002111, 55019111],
            [ 35002111, 59019111]])
    

# Make addition one-UseSum9 questions
def make_maths_s2_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg,  
        MathsToken.PLUS, QType.MATH_ADD, MathsBehavior.ADD_S2_TAG,
        [[ 55, 45],
        [ 45, 55],
        [ 45, 59],
        [ 35, 69],
        [ 25, 79],
        [ 15, 85],
        [ 15, 88],
        [ 15518, 14511],
        [ 14518, 15511],
        [ 24533, 25933],
        [ 23533, 26933],
        [ 32511, 7911],
        [ 31511, 8511],
        [ 551, 451],
        [ 451, 551],
        [ 10881, 41127],
        [ 41127, 10881],
        [ 12386, 82623],
        [ 108811, 411271],
        [ 411271, 108811],
        [ 123861, 826231],
        [ 994890, 80105],
        [ 970590, 96026],
        [ 994890, 80105],
        [ 970590, 96026],
        [ 1088111, 4112711],
        [ 4112711, 1088111],
        [ 1238611, 8262311],
        [ 10881111, 41127111],
        [ 41127111, 10881111],
        [ 12386111, 82623111]])
  

# These are two level UseSum9 cascades
def make_maths_s3_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg,  
        MathsToken.PLUS, QType.MATH_ADD, MathsBehavior.ADD_S3_TAG,
        [[ 555, 445],
        [ 3340, 6661],
        [ 8880, 1121],
        [ 1120, 8881],
        [ 123, 877],
        [ 877, 123],
        [ 321, 679],
        [ 679, 321],
        [ 1283, 78785]])
    

# These are three level UseSum9 cascades
def make_maths_s4_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg,  
        MathsToken.PLUS, QType.MATH_ADD, MathsBehavior.ADD_S4_TAG,
        [[ 5555, 4445],
        [ 55550, 44451],
        [ 3334, 6666],
        [ 33340, 66661],
        [ 8888, 1112],
        [ 88880, 11121],
        [ 1234, 8766],
        [ 4321, 5679]])


# These are four level UseSum9 cascades
def make_maths_s5_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg, 
        MathsToken.PLUS, QType.MATH_ADD, MathsBehavior.ADD_S5_TAG,
        [[ 44445, 55555],
        [ 33334, 66666],
        [ 88888, 11112],
        [ 12345, 87655],
        [ 54321, 45679],
        [ 45545, 54455],
        [ 36634, 63366],
        [ 81818, 18182],
        [ 87345, 12655],
        [ 55379, 44621]])
 

# Make M0 questions - when no column generates a Borrow One. Answer is always positive (or zero).
def make_maths_m0_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg, 
        MathsToken.MINUS, QType.MATH_SUB, MathsBehavior.SUB_M0_TAG,
        [[0, 0],
        [6, 6],
        [61, 60],
        [611, 600],
        [6111, 6000],
        [61111, 60000],
        [611111, 600000],
        [6111111, 6000000],
        [61111111, 60000000],
        [66666, 12345],
        [33333, 12321],
        [45762, 34551],
        [78901, 78901], # = +000000
        [23123, 23123], # = +000000
        [86, 15],
        [4440, 1230],
        [88746, 86544],
        [27833, 25133],
        [23533, 21133],
        [32501, 1],
        [31511, 1111],
        [55555, 12323],
        [45454, 22022],
        [66643, 3341],
        [66643, 30042],
        [99999, 44012],
        [61111, 30000],
        [99111, 99111], # = +000000
        [999991, 440120],
        [611111, 300000],
        [991111, 991111], # = +0000000
        [9999911, 4401200],
        [6111111, 3000000],
        [9911111, 9911111], # = +00000000
        [99999111, 44012000],
        [61111111, 30000000],
        [99111111, 99111111]]) # = +000000000


# Make subtraction M1 questions with exactly one "borrow 1" instance. Answer is always positive.
def make_maths_m1_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg, 
        MathsToken.MINUS, QType.MATH_SUB, MathsBehavior.SUB_M1_TAG,
        [[22222, 11113],
        [ 22222, 11131],
        [ 22222, 11311],
        [ 22222, 13111],
        [    14,     8],
        [   141,    80],
        [  1411,   800],
        [ 14111,  8000],
        [ 55514, 11108],
        [ 55141, 11080],
        [ 51411, 10800],
        [ 140111,  8000],
        [ 88888, 22229],
        [ 77777, 22292],
        [ 66666, 22922],
        [ 888888, 222292],
        [ 777777, 222922],
        [ 666666, 229222],
        [ 8888888, 2222922],
        [ 7777777, 2229222],
        [ 6666666, 2292222],
        [ 88888888, 22229222],
        [ 77777777, 22292222],
        [ 66666666, 22922222]])


# Make subtraction M2 questions containing BO and DZ. Answer is always positive (or zero).
def make_maths_m2_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg, 
        MathsToken.MINUS, QType.MATH_SUB, MathsBehavior.SUB_M2_TAG,
        [[22212, 11113],
        [ 22122, 11131],
        [ 21222, 11311],
        [   904,     8],
        [  9041,    80],
        [ 90411,   800],
        [ 55514, 11118],
        [ 55141, 11180],
        [ 51411, 11800],
        [ 88888, 22289],
        [ 77777, 22792],
        [ 66666, 26922],
        [ 888888, 222892],
        [ 777777, 227922],
        [ 666666, 269222],
        [ 8888888, 2228922],
        [ 7777777, 2279222],
        [ 6666666, 2692222],
        [ 88888888, 22289222],
        [ 77777777, 22792222],
        [ 66666666, 26922222]])


# Make subtraction M3,M4,... questions containing BO and multiple DZs. Answer is always positive (or zero).
def make_maths_m3_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg, 
        MathsToken.MINUS, QType.MATH_SUB, MathsBehavior.SUB_M3_TAG,
        [[22112, 11113],
        [ 21122, 11131],
        [ 99004,     8],
        [ 90041,    80],
        [ 55114, 11118],
        [ 51140, 11180],
        [ 88888, 22889],
        [ 87777, 27792],
        [ 888888, 228892],
        [ 877777, 277922],
        [ 8888888, 2288922],
        [ 7777777, 2779222],
        [ 88888888, 22889222],
        [ 77777777, 28892222]])
    

# Subtraction questions. Negative answer. Includes one BorrowOne. E.g. 100-200
def make_maths_n1_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg,  
        MathsToken.MINUS, QType.MATH_NEG, MathsBehavior.SUB_N1_TAG,
        [[1, 2],
        [4, 6],
        [10, 20],
        [100, 200],
        [400, 600],
        [1000, 2000],
        [4000, 6000],
        [10000, 20000],
        [40000, 60000],
        [100000, 200000]])


# Subtraction questions. Negative answer. Includes two BorrowOnes. E.g. 110-170
def make_maths_n2_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg,  
        MathsToken.MINUS, QType.MATH_NEG, MathsBehavior.SUB_N2_TAG,
        [[11, 13],
        [41, 47],
        [110, 130],
        [410, 470],
        [1100, 1300],
        [4100, 4700],
        [11000, 13000],
        [41000, 47000],
        [110000, 130000]])


# Subtraction questions. Negative answer. Includes three BorrowOnes. E.g. 111-117
def make_maths_n3_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg,  
    MathsToken.MINUS, QType.MATH_NEG, MathsBehavior.SUB_N3_TAG,
        [[111, 117],
        [432, 438],
        [1110, 1170],
        [4320, 4380],
        [11100, 11700],
        [43200, 43800],
        [111000, 117000]])


# Subtraction questions. Negative answer. Includes 4 BorrowOnes. E.g. 1111-1117
def make_maths_n4_questions_and_answers(cfg):
    return make_maths_questions_and_answers(cfg, 
        MathsToken.MINUS, QType.MATH_NEG, MathsBehavior.SUB_N4_TAG,
        [[3111, 3117],
        [3432, 3438],
        [31110, 31170],
        [34320, 34380],
        [311100, 311700],
        [343200, 343800],
        [3111000, 3117000]])



# Create a (matrix) batch of manually-curated mathematics test questions
def make_maths_test_questions_and_answers(cfg):

    # Start with a batch of random and manually-chosen questions
    varied_questions = maths_data_generator_core(cfg, MathsToken.PLUS if cfg.perc_add() > 0 else MathsToken.MINUS )

    if cfg.perc_add() > 0:
        varied_questions = torch.vstack((
            varied_questions,
            make_maths_s0_questions_and_answers(cfg),
            make_maths_s1_questions_and_answers(cfg),
            make_maths_s2_questions_and_answers(cfg),
            make_maths_s3_questions_and_answers(cfg),
            make_maths_s4_questions_and_answers(cfg),
            make_maths_s5_questions_and_answers(cfg)))

    if cfg.perc_sub > 0 :
        varied_questions = torch.vstack((
            varied_questions,
    
            # Subtraction questions with positive (or zero) answers
            make_maths_m0_questions_and_answers(cfg),  
            make_maths_m1_questions_and_answers(cfg),
            make_maths_m2_questions_and_answers(cfg),
            make_maths_m3_questions_and_answers(cfg),
            
            # Subtraction questions with negative answers
            make_maths_n1_questions_and_answers(cfg),         
            make_maths_n2_questions_and_answers(cfg),
            make_maths_n3_questions_and_answers(cfg),
            make_maths_n4_questions_and_answers(cfg)))
      
    return to_cuda(cfg, varied_questions)


# Test maths question prediction accuracy on the sample questions provided.
# Does NOT use acfg.* or UsefulInfo.* information 
# Used to estimate the accuracy of the model's predictions.
# Returns a reduced set of questions - removing questions that the model failed to answer.
def test_maths_questions_by_complexity(cfg, acfg, varied_questions):
        
    num_questions = varied_questions.shape[0]
    correct_list = [True] * num_questions
 
    all_logits = cfg.main_model(varied_questions.cuda())
    _, all_max_prob_tokens = logits_to_tokens_loss(cfg, all_logits, varied_questions.cuda())


    # Evaluate and categorize each object
    categorization_results = {}
    for question_num in range(num_questions):
        q_and_a = varied_questions[question_num]
    
        # Get the last cfg.num_answer_positions tokens in the q_and_a, which is the correct answer
        correct_answer_str = tokens_to_string(cfg, q_and_a[-cfg.num_answer_positions:])
        
        model_answer_str = tokens_to_string(cfg, all_max_prob_tokens[question_num])

        correct = (model_answer_str == correct_answer_str)
        correct_list[question_num] = correct

        major_tag, minor_tag = get_maths_question_complexity(cfg, q_and_a)     
        group_name = major_tag.value + "." + minor_tag.value

        if group_name not in categorization_results:
            categorization_results[group_name] = [0, 0]  # Initialize counts for new group

        if correct:
            categorization_results[group_name][0] += 1  # Increment good count for this group
        else:
            categorization_results[group_name][1] += 1  # Increment bad count for this group

        if acfg.show_test_failures and not correct:
            q_and_a_str = tokens_to_string(cfg, q_and_a)            
            print("Failed: Q&A:", q_and_a_str, "ModelAnswer:", model_answer_str, "Correct:", correct_answer_str, "Complexity:", group_name)


    # Calculate and print summary success rates per group
    acfg.num_varied_questions = 0
    acfg.num_varied_successes = 0
    for group_name, counts in categorization_results.items():
        total = sum(counts)
        success_rate = counts[0] / total * 100 if total != 0 else 0
        print(f"Group {group_name}: Success Rate = {success_rate:.2f}% ({counts[0]} good, {counts[1]} bad)")
        acfg.num_varied_questions += total
        acfg.num_varied_successes += counts[0]


    acfg.print_prediction_success_rate()
    if acfg.num_varied_successes < acfg.num_varied_questions:
        # Remove the questions that the model failed to answer as they turn up in every cell of the quanta maps
        org_size = varied_questions.shape[0]
        varied_questions = varied_questions[torch.tensor(correct_list)]
        new_size = varied_questions.shape[0]
        print("RESOLUTION: Understand these failures. Enrich the training data to provide more examples. Retrain the model.")
        print("INTERIM: Have reduced 'varied_questions' size from", org_size, "to", new_size, "so can continue.")
    
    return varied_questions


# Test accuracy of model in predicting question answers. Ablates all nodes at position
# Does NOT use UsefulInfo.* information. Used to populate UsefulInfo.useful_positions
def test_maths_questions_by_impact(cfg, acfg, questions, position : int, ablate : bool ):
    
    the_hooks = acfg.resid_put_hooks if ablate else None
    if ablate:
        assert not (the_hooks == None)
        
    acfg.ablate_node_locations = [NodeLocation(position, 0, True, 0)]  # Ablate all nodes at position
    all_losses_raw, all_max_prob_tokens = a_predict_questions(cfg, questions, the_hooks)

    num_fails = 0
    for question_num in range(questions.shape[0]):
        q = questions[question_num]
        assert q.shape[0] == cfg.n_ctx() # Check answer is embedded in question

        the_loss_mean = utils.to_numpy(loss_fn(all_losses_raw[question_num]).mean())
        
        # Only show the question if the loss exceeds the threshold (because of the ablated token position)
        if the_loss_mean > acfg.threshold:
       
            answer_str = tokens_to_string(cfg, all_max_prob_tokens[question_num])

            # Only count the question if the model got the question wrong
            impact_str = get_question_answer_impact(cfg, q, answer_str )
            if 'A' in impact_str:
                num_fails += 1

                if acfg.verbose :
                    print(tokens_to_string(cfg, q), "Q: ModelAnswer:", answer_str, "Impact:", impact_str, "Loss:", the_loss_mean )

    return num_fails


# Test accuracy of model in predicting question answers, when a single node is ablated. 
# Adds nodes to Useful.useful_nodes and adds tags to those nodes.
def test_maths_questions_and_add_useful_node_tags(cfg, acfg, questions, node_location, all_losses_raw, all_max_prob_tokens):
       
    num_fails = 0
    impact_fails = ""
    add_complexity_fails = ""
    sub_complexity_fails = ""
    neg_complexity_fails = ""

    for question_num in range(questions.shape[0]):
        q = questions[question_num]

        the_loss_mean = utils.to_numpy(loss_fn(all_losses_raw[question_num]).mean())

        # Only show the question if the loss exceeds the threshold (because of the ablated token position)
        if the_loss_mean > acfg.threshold:
            answer_str = tokens_to_string(cfg, all_max_prob_tokens[question_num])

            impact_str = get_question_answer_impact(cfg, q, answer_str )
            # Only count the question if the model got the question wrong
            if 'A' in impact_str:
                num_fails += 1

                impact_fails += impact_str

                major_tag, minor_tag = get_maths_question_complexity(cfg, q)
                if major_tag == QType.MATH_ADD:
                    add_complexity_fails += minor_tag.value
                elif major_tag == QType.MATH_SUB:
                    sub_complexity_fails += minor_tag.value
                elif major_tag == QType.MATH_NEG:
                    neg_complexity_fails += minor_tag.value
                    
                if acfg.verbose :
                    print(tokens_to_string(cfg, q), "U: ModelAnswer:", answer_str, "Complexity:", major_tag, "Impact:", impact_str, "Loss:", the_loss_mean )

    if num_fails > 0:

        # Add percentage failure quanta
        perc = int( 100.0 * num_fails / len(questions))
        cfg.add_useful_node_tag( node_location, QType.FAIL.value, str(perc) )

        # Add summary of all answer digit impact quanta failures
        cfg.add_useful_node_tag( node_location, QType.IMPACT.value, "A" + sort_unique_digits(impact_fails, True) )

        # Add summary of all addition question complexity quanta failures
        if add_complexity_fails != "":
            cfg.add_useful_node_tag( node_location, QType.MATH_ADD.value, "S" + sort_unique_digits(add_complexity_fails, False) )

        # Add summary of all subtraction question complexity quanta failures
        if sub_complexity_fails != "":
            cfg.add_useful_node_tag( node_location, QType.MATH_SUB.value, "M" + sort_unique_digits(sub_complexity_fails, False) )
        if neg_complexity_fails != "":
            cfg.add_useful_node_tag( node_location, QType.MATH_NEG.value, "N" + sort_unique_digits(neg_complexity_fails, False) )
          

TRICASE_QUESTIONS = 100


def make_tricase_questions(cfg, test_digit, test_case, operation):
    limit = 10 ** test_digit
    questions = []
    for i in range(TRICASE_QUESTIONS):
        x_noise = 0
        y_noise = 0

        if operation == MathsToken.PLUS:
            if test_case == 8:
                # These are n_digit addition questions where x and y sum is between 0 to 8
                x = random.randint(0, 8)
                y = random.randint(0, 8-x)
            if test_case == 9:
                # These are n_digit addition questions where x and y sum is 9
                x = random.randint(0, 9)
                y = 9 - x
            if test_case == 10:
                # These are n_digit addition questions where x and y sum is between 10 to 18
                x = random.randint(1, 9)
                y = random.randint(10-x, 9)

            # Randomise the lower digits - ensuring that x_noise + y_noise dont cause a MakeCarry
            x_noise = random.randint(0, limit-1)
            y_noise = random.randint(0, limit-1 - x_noise)


        if operation == MathsToken.MINUS:
            if test_case == 8:
                # These are n_digit subtraction questions where x - y < 0
                x = random.randint(0, 8)
                y = random.randint(x+1, 9)
            if test_case == 9:
                # These are n_digit subtraction questions where x - y is 0
                x = random.randint(0, 9)
                y = x
            if test_case == 10:
                # These are n_digit subtraction questions where x - y > 0
                x = random.randint(1, 9)
                y = random.randint(0, x-1)

            # Randomise the lower digits - ensuring that x_noise + y_noise dont cause a BorrowOne
            x_noise = random.randint(0, limit-1)
            y_noise = random.randint(0, x_noise)


        x = x * limit + x_noise
        y = y * limit + y_noise
        questions.append([x, y])

    qtype = QType.MATH_ADD if operation == MathsToken.PLUS else QType.MATH_SUB # Inaccurate. Could be QType.MATH_NEG
    return make_maths_questions_and_answers(cfg, operation, qtype, MathsBehavior.UNKNOWN, questions)



def make_maths_tricase_questions_core(cfg, test_digit, operation):
    q1 = make_tricase_questions(cfg, test_digit, 8, operation)
    q2 = make_tricase_questions(cfg, test_digit, 9, operation)
    q3 = make_tricase_questions(cfg, test_digit, 10, operation)

    return torch.vstack((q1, q2, q3))


# Create a cache of sample (matrix) maths questions based on the T8, T9, T10 categorisation
def make_maths_tricase_questions(cfg):
    cfg.tricase_questions_dict = {}
    for answer_digit in range(cfg.n_digits):
        for operation in [MathsToken.PLUS, MathsToken.MINUS]:
            t_questions = make_maths_tricase_questions_core(cfg, answer_digit, operation)
            # Use a tuple of (answer_digit, operation) as the key for indexing
            cfg.tricase_questions_dict[(answer_digit, operation)] = t_questions


def test_correctness_on_num_questions(cfg, num_questions=1000000):
  store_perc_sub = cfg.perc_sub
  store_perc_mult = cfg.perc_mult

  def print_config():
      print("%Mult=", cfg.perc_mult, "%Sub=", cfg.perc_sub, "%Add=", cfg.perc_add(), "File", cfg.file_config_prefix())

  print_config()
  print()

  if cfg.perc_add() > 0:
    print("Addition:")
    cfg.perc_sub = 0
    cfg.perc_mult = 0
    test_correctness_on_num_questions_core(cfg, num_questions=num_questions)

  if store_perc_sub > 0:
    print("Subtraction:")
    cfg.perc_sub = 100
    cfg.perc_mult = 0
    test_correctness_on_num_questions_core(cfg, num_questions=num_questions)
    print()

  cfg.perc_sub = store_perc_sub
  cfg.perc_mult = store_perc_mult


def test_correctness_on_num_questions_core(cfg, num_questions=1000000):
  acfg.verbose = False

  cfg.analysis_seed = 345621  # Randomly chosen
  local_ds = maths_data_generator(cfg=cfg)  # Re-initialise the data generator

  the_successes = 0
  the_fails = 0

  num_batches = num_questions//cfg.batch_size
  for epoch in tqdm(range(num_batches)):
      tokens = next(local_ds)

      the_fails = test_maths_questions_by_impact(cfg, acfg, tokens, 0, False)

      if the_fails>0:
        break

      the_successes = the_successes + cfg.batch_size

      if epoch % 100 == 0:
          print("Batch", epoch, "of", num_batches, "#Successes=", the_successes)

  print("successes", the_successes, "num_fails", the_fails)
  if the_fails > 0:
    "WARNING: Model is not fully accurate. It failed the 1M Q test"
