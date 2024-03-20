import random
import torch
import transformer_lens.utils as utils

from .model_loss import logits_to_tokens_loss, loss_fn
from .model_token_to_char import tokens_to_string

from .useful_node import NodeLocation

from .quanta_type import QuantaType
from .quanta_map_impact import get_question_answer_impact, sort_unique_digits

from .ablate_hooks import a_predict_questions

from .maths_vocab import MathsTokens
from .maths_tag import MathsBehavior
from .maths_data_generator import maths_data_generator_core, make_maths_questions_and_answers
from .maths_utilities import tokens_to_answer
from .maths_complexity import get_maths_question_complexity


# Create a (matrix) batch of manually-curated mathematics test questions
def make_maths_test_questions_and_answers(cfg):

    # Start with a batch of random and manually-chosen questions
    varied_questions = maths_data_generator_core(cfg, MathsTokens.PLUS if cfg.perc_add() > 0 else MathsTokens.MINUS )

    if cfg.perc_add() > 0:
        varied_questions = torch.vstack((
        varied_questions.cuda(),
            # Make BaseAdd questions
            make_maths_questions_and_answers(cfg, MathsTokens.PLUS, QuantaType.MATH_ADD, MathsBehavior.ADD_S0_TAG,
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
              [10000000, 2111111]]).cuda(),
            # Make UseCarry1 (addition) questions
            make_maths_questions_and_answers(cfg, MathsTokens.PLUS, QuantaType.MATH_ADD, MathsBehavior.ADD_S1_TAG,
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
              [ 35002111, 59019111]]).cuda(),
            # Make SimpleUseSum9 (addition) questions
            make_maths_questions_and_answers(cfg, MathsTokens.PLUS, QuantaType.MATH_ADD, MathsBehavior.ADD_S2_TAG,
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
              [ 12386111, 82623111]]).cuda(),
            # These are two level UseSum9 cascades
            make_maths_questions_and_answers(cfg, MathsTokens.PLUS, QuantaType.MATH_ADD, MathsBehavior.ADD_S3_TAG,
              [[ 555, 445],
              [ 3340, 6661],
              [ 8880, 1121],
              [ 1120, 8881],
              [ 123, 877],
              [ 877, 123],
              [ 321, 679],
              [ 679, 321],
              [ 1283, 78785]]).cuda(),
            # These are three level UseSum9 cascades
            make_maths_questions_and_answers(cfg, MathsTokens.PLUS, QuantaType.MATH_ADD, MathsBehavior.ADD_S4_TAG,
              [[ 5555, 4445],
              [ 55550, 44451],
              [ 3334, 6666],
              [ 33340, 66661],
              [ 8888, 1112],
              [ 88880, 11121],
              [ 1234, 8766],
              [ 4321, 5679]]).cuda(),
            # These are four level UseSum9 cascades
            make_maths_questions_and_answers(cfg, MathsTokens.PLUS, QuantaType.MATH_ADD, MathsBehavior.ADD_S5_TAG,
              [[ 44445, 55555],
              [ 33334, 66666],
              [ 88888, 11112],
              [ 12345, 87655],
              [ 54321, 45679],
              [ 45545, 54455],
              [ 36634, 63366],
              [ 81818, 18182],
              [ 87345, 12655],
              [ 55379, 44621]]).cuda(),
            # Make questions focus mainly on 1 digit at a time
            # (assuming that the 0 + 0 digit additions/subtractions are trivial bigrams)
            make_maths_questions_and_answers(cfg, MathsTokens.PLUS, QuantaType.MATH_ADD, "",
              [[ 1, 0],
              [ 4, 3],
              [ 5, 5],
              [ 8, 1],
              [ 40, 31],
              [ 44, 46],
              [ 400, 311],
              [ 440, 461],
              [ 800, 111],
              [ 270, 471],
              [ 600, 311],
              [ 4000, 3111],
              [ 4400, 4611],
              [ 6000, 3111],
              [ 7000, 4111],
              [ 40000, 31111],
              [ 44000, 45111],
              [ 60000, 31111],
              [ 70000, 41111],
              [ 10000, 21111],
              [ 15000, 25111],
              [ 35000, 35111],
              [ 45000, 85111],
              [ 67000, 85111],
              [ 99000, 76111],
              [ 76000, 99111],
              [ 670000, 851111],
              [ 990000, 761111],
              [ 760000, 991111],
              [ 6700000, 8511111],
              [ 9900000, 7611111],
              [ 7600000, 9911111],
              [ 67000000, 85111111],
              [ 99000000, 76111111],
              [ 76000000, 99111111]]).cuda()))
  

    if cfg.perc_sub > 0 :
        varied_questions = torch.vstack((
            varied_questions.cuda(),
            # Make M0 questions - when no column generates a Borrow One. Answer is always positive (or zero).
            make_maths_questions_and_answers(cfg, MathsTokens.MINUS, QuantaType.MATH_SUB, MathsBehavior.SUB_S0_TAG,
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
              [99111111, 99111111]]).cuda(), # = +000000000
            # Make subtraction M1 questions with exactly one "borrow 1" instance. Answer is always positive.
            make_maths_questions_and_answers(cfg, MathsTokens.MINUS, QuantaType.MATH_SUB, MathsBehavior.SUB_S1_TAG,
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
              [ 66666666, 22922222]]).cuda(),
            # Make subtraction M2 questions containing BO and DZ. Answer is always positive (or zero).
            make_maths_questions_and_answers(cfg, MathsTokens.MINUS, QuantaType.MATH_SUB, MathsBehavior.SUB_S2_TAG,
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
              [ 66666666, 26922222]]).cuda(),
            # Make subtraction M3,M4,... questions containing BO and multiple DZs. Answer is always positive (or zero).
            make_maths_questions_and_answers(cfg, MathsTokens.MINUS, QuantaType.MATH_SUB, MathsBehavior.SUB_S3_TAG,
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
              [ 77777777, 28892222]]).cuda(),
            # Make subtraction questions with negative answers
            make_maths_questions_and_answers(cfg, MathsTokens.MINUS, QuantaType.MATH_SUB, MathsBehavior.SUB_NG_TAG,
              [[0, 1],
              [7, 9],
              [12345, 33333],
              [888, 11111],
              [2362, 23123],
              [15, 81],
              [1111, 4440],
              [24033, 25133],
              [23533, 88133],
              [5511, 12323],
              [4511, 22209],
              [ 88888, 88889],
              [ 55555, 55556],
              [ 88881, 88891],
              [ 55551, 55561],
              [ 88811, 88911],
              [ 55511, 55611],
              [ 88746, 89544],
              [ 27833, 29133],
              [ 23533, 23833],
              [ 31511, 41111],
              [ 55555, 62323],
              [ 45454, 72022],
              [ 66643, 73341],
              [ 66643, 90042],
              [ 99998, 99999],
              [ 8, 12],
              [ 41, 232],
              [ 44, 523],
              [ 234, 334],
              [ 7777, 8434],
              [ 88888, 92222],
              [ 77777, 84340],
              [ 888888, 922220],
              [ 777777, 843400],
              [ 8888888, 9222200],
              [ 7777777, 8434000],
              [ 88888888, 92222000],
              [ 77777777, 84340000]]).cuda()))
      
    return varied_questions


# Test maths question prediction accuracy on the sample questions provided.
# Does NOT use acfg.* or UsefulInfo.* information 
# Used to estimate the accuracy of the model's predictions.
# Returns a reduced set of questions - removing questions that the model failed to answer.
def test_maths_questions_by_complexity(cfg, acfg, varied_questions):
        
    num_questions = varied_questions.shape[0]
    correct_list = [True] * num_questions
    print( "PQR1", num_questions)

    all_logits = cfg.main_model(varied_questions.cuda())
    _, all_max_prob_tokens = logits_to_tokens_loss(cfg, all_logits, varied_questions.cuda())

    # Evaluate and categorize each object
    categorization_results = {}
    for question_num in range(num_questions):
        q = varied_questions[question_num]
        print( "PQR2", q.shape )
    
        model_answer_str = tokens_to_string(cfg, all_max_prob_tokens[question_num])
        model_answer_num = int(model_answer_str)

        major_tag, minor_tag = get_maths_question_complexity(cfg, q)
        group_name = major_tag + "." + minor_tag

        correct_answer = tokens_to_string(cfg, q)
        correct = (model_answer_num == correct_answer)
        correct_list[question_num] = correct

        if group_name not in categorization_results:
            categorization_results[group_name] = [0, 0]  # Initialize counts for new group

        if correct:
            categorization_results[group_name][0] += 1  # Increment good count for this group
        else:
            categorization_results[group_name][1] += 1  # Increment bad count for this group

        if acfg.show_test_failures and not correct:
            print("Failed: ModelAnswer:", model_answer_str, "Correct:", correct_answer, "Complexity:", group_name)

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
    
    print( "PQR1", questions.shape)
        
    acfg.ablate_node_locations = [NodeLocation(position, 0, True, 0)]  # Ablate all nodes at position
    all_losses_raw, all_max_prob_tokens = a_predict_questions(cfg, questions, the_hooks)

    num_fails = 0
    for question_num in range(questions.shape[0]):
        q = questions[question_num]
        print( "PQR2", q.shape[0])
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
def test_maths_questions_and_add_useful_node_tags(cfg, acfg, node_location, questions, the_hooks):
       
    acfg.ablate_node_locations = [node_location]  # Ablate this node  
    all_losses_raw, all_max_prob_tokens = a_predict_questions(cfg, questions, the_hooks)

    num_fails = 0
    impact_fails = ""
    add_complexity_fails = ""
    sub_complexity_fails = ""

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
                if major_tag == QuantaType.MATH_ADD:
                    add_complexity_fails += minor_tag
                elif major_tag == QuantaType.MATH_SUB:
                    sub_complexity_fails += minor_tag

                if acfg.verbose :
                    print(tokens_to_string(cfg, q), "U: ModelAnswer:", answer_str, "Complexity:", major_tag, "Impact:", impact_str, "Loss:", the_loss_mean )

    if num_fails > 0:

        # Add percentage failure quanta
        perc = int( 100.0 * num_fails / len(questions))
        cfg.add_useful_node_tag( node_location, QuantaType.FAIL, str(perc) )

        # Add summary of all answer digit impact quanta failures
        cfg.add_useful_node_tag( node_location, QuantaType.IMPACT, "A" + sort_unique_digits(impact_fails, True) )

        # Add summary of all addition question complexity quanta failures
        if add_complexity_fails != "":
            cfg.add_useful_node_tag( node_location, QuantaType.MATH_ADD, "S" + sort_unique_digits(add_complexity_fails, False) )

        # Add summary of all subtraction question complexity quanta failures
        if sub_complexity_fails != "":
            sub_complexity_fails = sort_unique_digits(sub_complexity_fails, False)
            if sub_complexity_fails == "":
                sub_complexity_fails = MathsBehavior.SUB_NG_TAG
            else:
                sub_complexity_fails = "M" + sub_complexity_fails
            cfg.add_useful_node_tag( node_location, QuantaType.MATH_SUB, sub_complexity_fails )
          

TRICASE_QUESTIONS = 100


def make_tricase_questions(cfg, test_digit, test_case, operation):
    limit = 10 ** test_digit
    questions = []
    for i in range(TRICASE_QUESTIONS):
        x_noise = 0
        y_noise = 0

        if operation == MathsTokens.PLUS:
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


        if operation == MathsTokens.MINUS:
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

    return make_maths_questions_and_answers(cfg, operation, "", "", questions)



def make_maths_tricase_questions_core(cfg, test_digit, operation):
    q1 = make_tricase_questions(cfg, test_digit, 8, operation)
    q2 = make_tricase_questions(cfg, test_digit, 9, operation)
    q3 = make_tricase_questions(cfg, test_digit, 10, operation)

    return torch.vstack((q1, q2, q3))


# Create a cache of sample (matrix) maths questions based on the T8, T9, T10 categorisation
def make_maths_tricase_questions(cfg):
    cfg.tricase_questions_dict = {}
    for answer_digit in range(cfg.n_digits):
        for operation in [MathsTokens.PLUS, MathsTokens.MINUS]:
            t_questions = make_maths_tricase_questions_core(cfg, answer_digit, operation)
            # Use a tuple of (answer_digit, operation) as the key for indexing
            cfg.tricase_questions_dict[(answer_digit, operation)] = t_questions