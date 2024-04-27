import torch

from QuantaTools.quanta_constants import QType

from QuantaTools.maths_tools.maths_constants import MathsToken, MathsBehavior
from QuantaTools.maths_tools.maths_data_generator import maths_data_generator_core, make_maths_questions_and_answers


# Move the data to the GPU (if any) for faster processing
def to_cuda(cfg, data):
    if cfg.use_cuda:
        return data.cuda()
    return data    



def make_maths_s0_questions_and_answers(cfg):
    # Make addition BaseAdd questions
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


def make_maths_s1_questions_and_answers(cfg):
    # Make addition UseCarry1 questions
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


def make_maths_s2_questions_and_answers(cfg):
    # Make addition one-UseSum9 questions
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


def make_maths_s3_questions_and_answers(cfg):
    # These are two level UseSum9 cascades
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


def make_maths_s4_questions_and_answers(cfg):
    # These are three level UseSum9 cascades
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


def make_maths_s5_questions_and_answers(cfg):
    # These are four level UseSum9 cascades
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



def make_maths_m0_questions_and_answers(cfg):
    # Make M0 questions - when no column generates a Borrow One. Answer is always positive (or zero).
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


def make_maths_m1_questions_and_answers(cfg):
    # Make subtraction M1 questions with exactly one "borrow 1" instance. Answer is always positive.
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


def make_maths_m2_questions_and_answers(cfg):
    # Make subtraction M2 questions containing BO and DZ. Answer is always positive (or zero).
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


def make_maths_m3_questions_and_answers(cfg):
    # Make subtraction M3,M4,... questions containing BO and multiple DZs. Answer is always positive (or zero).
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


def make_maths_n1_questions_and_answers(cfg):
    # Subtraction questions. Negative answer. Includes one BorrowOne. E.g. 100-200
    return make_maths_questions_and_answers(cfg,
        MathsToken.MINUS, QType.MATH_NEG, MathsBehavior.NEG_N1_TAG,
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


def make_maths_n2_questions_and_answers(cfg):
    # Subtraction questions. Negative answer. Includes two BorrowOnes. E.g. 110-170
    return make_maths_questions_and_answers(cfg,
        MathsToken.MINUS, QType.MATH_NEG, MathsBehavior.NEG_N2_TAG,
        [[11, 13],
        [41, 47],
        [110, 130],
        [410, 470],
        [1100, 1300],
        [4100, 4700],
        [11000, 13000],
        [41000, 47000],
        [110000, 130000]])


def make_maths_n3_questions_and_answers(cfg):
    # Subtraction questions. Negative answer. Includes three BorrowOnes. E.g. 111-117
    return make_maths_questions_and_answers(cfg,
    MathsToken.MINUS, QType.MATH_NEG, MathsBehavior.NEG_N3_TAG,
        [[111, 117],
        [432, 438],
        [1110, 1170],
        [4320, 4380],
        [11100, 11700],
        [43200, 43800],
        [111000, 117000]])


def make_maths_n4_questions_and_answers(cfg):
    # Subtraction questions. Negative answer. Includes 4 BorrowOnes. E.g. 1111-1117
    return make_maths_questions_and_answers(cfg,
        MathsToken.MINUS, QType.MATH_NEG, MathsBehavior.NEG_N4_TAG,
        [[3111, 3117],
        [3432, 3438],
        [31110, 31170],
        [34320, 34380],
        [311100, 311700],
        [343200, 343800],
        [3111000, 3117000]])



def make_maths_test_questions_and_answers(cfg):
    # Create a (matrix) batch of manually-curated mathematics test questions

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


