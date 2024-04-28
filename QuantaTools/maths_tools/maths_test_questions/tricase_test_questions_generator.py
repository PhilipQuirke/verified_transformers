from dataclasses import dataclass
from typing import Tuple
import random

import torch

from QuantaTools.quanta_constants import QType
from QuantaTools.maths_tools.maths_constants import MathsBehavior, MathsTask, MathsToken
from QuantaTools.maths_tools.maths_data_generator import make_maths_questions_and_answers


# Create a cache of sample (matrix) maths questions based on the T8, T9, T10 categorisation
EACH_CASE_TRICASE_QUESTIONS = 100
TOTAL_TRICASE_QUESTIONS = 3*EACH_CASE_TRICASE_QUESTIONS

@dataclass
class OperatorQTypeNumber:
    operator: MathsToken
    qtype:QType
    number: int

@dataclass
class CustomTriclassConfig:
    operators_qtypes_counts: Tuple[OperatorQTypeNumber] = (
        OperatorQTypeNumber(MathsToken.PLUS, QType.MATH_ADD, TOTAL_TRICASE_QUESTIONS),
        OperatorQTypeNumber(MathsToken.MINUS, QType.MATH_SUB, TOTAL_TRICASE_QUESTIONS),
        OperatorQTypeNumber(MathsToken.MINUS, QType.MATH_NEG, TOTAL_TRICASE_QUESTIONS)
    )

def make_tricase_questions(
        cfg, test_digit: int, test_case: int, operation: MathsToken, num_questions=TOTAL_TRICASE_QUESTIONS, qtype: QType = None,
        make_borrow: str = "never"
):
    """
    Returns a set of questions over a number of test digits, given an operation and optionally a qtype
    make_borrow should be "never", "always" or "mixed" and controls whether we allow for
    borrow ones in less significant digits or not.
    make_borrow will be overidden if the case_type and qtype require it, such as for
    tricase test_case=8 and qtype=Math_NEG
    """
    assert make_borrow in ["always", "never", "mixed"]
    limit = 10 ** test_digit
    questions = []
    assert qtype in [None, QType.MATH_SUB, QType.MATH_NEG, QType.UNKNOWN], f"Qtype must be none, sub, neg or unknown"
    assert test_case in [8,9,10], f"Tricase test cases must be 8,9 or 10, received {test_case}"
    assert operation in [MathsToken.PLUS, MathsToken.MINUS], f"Tricase operation must be in [plus,minus]={[MathsToken.PLUS, MathsToken.MINUS]}, recieved operation {operation}"

    for i in range(num_questions):
        x_noise = 0
        y_noise = 0

        if operation == MathsToken.PLUS:
            assert qtype not in [QType.MATH_NEG, QType.MATH_NEG], f"Negative qtypes not supported for PLUS operator, received {qtype}"
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

        def make_noise(make_borrow="never"):
            if make_borrow == "never":
                x_noise = random.randint(0, limit - 1)
                y_noise = random.randint(0, x_noise)
            elif make_borrow == "always":
                y_noise = random.randint(1, limit - 1)
                x_noise = random.randint(0, y_noise-1)
            else:
                x_noise = random.randint(0, limit - 1)
                y_noise = random.randint(0, limit - 1)
            return x_noise, y_noise

        # Check digit positions for viability
        if operation == MathsToken.MINUS:
            if test_case == 8:
                # These are n_digit subtraction questions where x - y < 0
                x = random.randint(0, 8)
                y = random.randint(x+1, 9)
                if qtype == QType.MATH_SUB:
                    raise Exception(f'Cannot have both Math_Sub and test_case 8 true simultaneously.')
                else:
                    x_noise, y_noise = make_noise(make_borrow=make_borrow)

            # These are n_digit subtraction questions where x - y is 0
            if test_case == 9 and test_digit == cfg.n_digits and qtype==QType.MATH_NEG:
                raise Exception("Test Case 9 is not possible for Math_NEG on last digit")

            if test_case == 9:
                # These are n_digit subtraction questions where x - y is 0
                x = random.randint(0, 9)
                y = x
                if qtype == QType.MATH_NEG:
                    local_make_borrow = "always"
                elif qtype == QType.MATH_SUB:
                    local_make_borrow = "never"
                else:
                    local_make_borrow=make_borrow
                x_noise, y_noise = make_noise(local_make_borrow)

            if test_case == 10:
                # These are n_digit subtraction questions where x - y > 0
                x = random.randint(1, 9)
                y = random.randint(0, x-1)

                if qtype == QType.MATH_NEG:
                    raise Exception(f'Cannot have both Math_Neg and test_case 10 true simultaneously.')
                else:
                    x_noise, y_noise = make_noise(make_borrow)


        x = x * limit + x_noise
        y = y * limit + y_noise
        questions.append([x, y])

    if qtype is not None:  # We have enforced qtype remains consistent with questions returned
        return make_maths_questions_and_answers(cfg, operation, qtype, MathsBehavior.UNKNOWN, questions)

    elif operation == MathsToken.PLUS:  # qtype not relevant for MathsToken.PLUS
        qtype = QType.MATH_ADD #if operation == MathsToken.PLUS else QType.MATH_SUB # Inaccurate. Will be a mix of QType.MATH_SUB and QType.MATH_NEG

        return make_maths_questions_and_answers(cfg, operation, qtype, MathsBehavior.UNKNOWN, questions)

    elif operation == MathsToken.MINUS:
        sub_questions = [question for question in questions if question[0] >= question[1]]

        sub_question_tensors = make_maths_questions_and_answers(cfg, operation, QType.MATH_SUB, MathsBehavior.UNKNOWN, sub_questions)

        neg_questions = [question for question in questions if question[0] < question[1]]

        neg_question_tensors = make_maths_questions_and_answers(cfg, operation, QType.MATH_NEG, MathsBehavior.UNKNOWN, neg_questions)
        print(f"Created {len(sub_question_tensors)} MATH_SUB questions for MINUS and {len(neg_question_tensors)} MATH_NEG questions.")

        return torch.vstack([sub_question_tensors, neg_question_tensors])

    else:
        raise Exception(f"Unsupported operation {operation}.")

def make_maths_tricase_questions_core(cfg, test_digit, operation, num_questions=TOTAL_TRICASE_QUESTIONS):
    assert num_questions%3==0, "Number of questions must be divisible by 3"
    local_num_questions = int(num_questions/3)
    q1 = make_tricase_questions(cfg, test_digit, 8, operation, num_questions=local_num_questions)
    q2 = make_tricase_questions(cfg, test_digit, 9, operation, num_questions=local_num_questions)
    q3 = make_tricase_questions(cfg, test_digit, 10, operation, num_questions=local_num_questions)

    return torch.vstack((q1, q2, q3))

def make_maths_tricase_questions(cfg, num_questions=TOTAL_TRICASE_QUESTIONS):
    cfg.tricase_questions_dict = {}
    for answer_digit in range(cfg.n_digits):
        for operation in [MathsToken.PLUS, MathsToken.MINUS]:
            t_questions = make_maths_tricase_questions_core(cfg, answer_digit, operation, num_questions=num_questions)
            # Use a tuple of (answer_digit, operation) as the key for indexing
            cfg.tricase_questions_dict[(answer_digit, operation)] = t_questions

def make_maths_tricase_questions_customized(cfg, custom_triclass_config=CustomTriclassConfig()):
    """
    Creates a dictionary of tricase questions in cfg.tricase_questions_dict.
    This dictionary is indexed by (answer_digit, operator, question_type) with custom_triclass_config.number examples of each.
    """
    cfg.tricase_questions_dict = {}
    for answer_digit in range(cfg.n_digits):
        operators_qtype_numbers = custom_triclass_config.operators_qtypes_counts

        for operator_qtype_number in operators_qtype_numbers:
            num_questions = operator_qtype_number.number
            qtype = operator_qtype_number.qtype
            operator = operator_qtype_number.operator

            if qtype in [QType.MATH_NEG, QType.MATH_SUB] and operator == MathsToken.PLUS:
                raise Exception(f'A qtype of MATH_NEG or MATH_SUB is not supported with plus operator.')

            if qtype == QType.MATH_SUB:
                # Only test cases 9 and 10 are supported for MATH_SUB
                target_cases = [9, 10]
                local_num_questions = int(num_questions / len(target_cases))
                all_questions = [make_tricase_questions(
                    cfg, test_digit=answer_digit, test_case=test_case, operation=operator, qtype=qtype, num_questions=local_num_questions
                ) for test_case in target_cases]
            elif qtype == QType.MATH_NEG:
                # Only test cases 8 and 9 are supported for MATH_SUB
                target_cases = [8, 9]
                local_num_questions = int(num_questions / len(target_cases))
                all_questions = [make_tricase_questions(
                    cfg, test_digit=answer_digit, test_case=test_case, operation=operator, qtype=qtype,
                    num_questions=local_num_questions
                ) for test_case in target_cases]
            elif qtype == QType.MATH_ADD:
                target_cases = [8, 9, 10]
                local_num_questions = int(num_questions / len(target_cases))
                all_questions = [make_tricase_questions(
                    cfg, test_digit=answer_digit, test_case=test_case, operation=operator, qtype=qtype,
                    num_questions=local_num_questions
                ) for test_case in target_cases]
            else:
                target_cases = [8, 9, 10]
                local_num_questions = int(num_questions / len(target_cases))
                all_questions = [make_tricase_questions(
                    cfg, test_digit=answer_digit, test_case=test_case, operation=operator, qtype=qtype,
                    num_questions=local_num_questions
                ) for test_case in target_cases]

            num_questions_created = sum([len(questions) for questions in all_questions])
            assert num_questions_created == num_questions, f"Created {num_questions_created} when requested with {num_questions} for {operator_qtype_number}"
            cfg.tricase_questions_dict[(answer_digit, operator, qtype)] = torch.vstack(all_questions)
