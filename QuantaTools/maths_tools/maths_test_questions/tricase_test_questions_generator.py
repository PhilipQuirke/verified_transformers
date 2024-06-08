from dataclasses import dataclass
from typing import Tuple
import random

import torch

from QuantaTools.quanta_constants import QType
from QuantaTools.maths_tools.maths_constants import MathsBehavior, MathsTask, MathsToken, TriCaseBehavior, maths_tokens_to_names

from QuantaTools.maths_tools.maths_data_generator import make_maths_questions_and_answers


# Create a cache of sample (matrix) maths questions based on the ST8, ST9, ST10 categorisation
EACH_CASE_TRICASE_QUESTIONS = 100
TOTAL_TRICASE_QUESTIONS = 3*EACH_CASE_TRICASE_QUESTIONS

@dataclass(eq=True, order=True, frozen=True)
class OperatorQTypeNumber:
    operator: MathsToken
    qtype:QType
    number: int

@dataclass(eq=True, order=True, frozen=True)
class DigitOperatorQTypeTricase:
    digit: int
    operator: MathsToken
    qtype:QType
    test_case: TriCaseBehavior

@dataclass
class CustomTriclassConfig:
    operators_qtypes_counts: Tuple[OperatorQTypeNumber] = (
        OperatorQTypeNumber(MathsToken.PLUS, QType.MATH_ADD, TOTAL_TRICASE_QUESTIONS),
        OperatorQTypeNumber(MathsToken.MINUS, QType.MATH_SUB, TOTAL_TRICASE_QUESTIONS),
        OperatorQTypeNumber(MathsToken.MINUS, QType.MATH_NEG, TOTAL_TRICASE_QUESTIONS)
    )

def pad_small_set_of_questions(cfg, sample_pairs_of_numbers: list, target_number: int, digit: int):
    """
    Sample a value and add it to list
    """
    unique_pairs_set = set(sample_pairs_of_numbers)
    unique_pairs_list = list(unique_pairs_set)
    attempts = 0

    while len(unique_pairs_set) < target_number and attempts < 2*target_number:
        attempts += 1
        num_digits = cfg.n_digits - digit - 1
        # Find a random number, and offset it so it doesn't affect test digit.
        random_addition = 10**(digit+1) * random.randint(0, 10**num_digits)
        random_choice = random.choice(unique_pairs_list)
        new_choice = (random_choice[0] + random_addition, random_choice[1] + random_addition)
        unique_pairs_set.add(new_choice)

    return list(unique_pairs_set)

def make_single_tricase_question(
        cfg, test_digit: int, test_case: TriCaseBehavior, operation: MathsToken,
        qtype: QType = None, make_borrow: str = "mixed"
):
    x_noise = 0
    y_noise = 0
    limit = 10 ** test_digit

    def make_noise(make_borrow="never"):
        if make_borrow == "never":
            x_noise = random.randint(0, limit - 1)
            y_noise = random.randint(0, x_noise)
        elif make_borrow == "always":
            y_noise = random.randint(1, limit - 1)
            x_noise = random.randint(0, y_noise - 1)
        else:
            x_noise = random.randint(0, limit - 1)
            y_noise = random.randint(0, limit - 1)
        return x_noise, y_noise

    if operation == MathsToken.PLUS:
        if test_case == TriCaseBehavior.ST8:
            # These are n_digit addition questions where x and y sum is between 0 to 8
            x = random.randint(0, 8)
            y = random.randint(0, 8 - x)
        elif test_case == TriCaseBehavior.ST9:
            # These are n_digit addition questions where x and y sum is 9
            x = random.randint(0, 9)
            y = 9 - x
        elif test_case == TriCaseBehavior.ST10:
            # These are n_digit addition questions where x and y sum is between 10 to 18
            x = random.randint(1, 9)
            y = random.randint(10 - x, 9)
        else:
            raise Exception(f'Expected test case in ST8, ST9 or ST10. Received {test_case}')

        # Randomise the lower digits - ensuring that x_noise + y_noise dont cause a MakeCarry
        x_noise = random.randint(0, limit - 1)
        y_noise = random.randint(0, limit - 1 - x_noise)


    # Check digit positions for viability
    elif operation == MathsToken.MINUS:
        if test_case == TriCaseBehavior.MT3:
            # These are n_digit subtraction questions where x - y < 0
            x = random.randint(0, 8)
            y = random.randint(x + 1, 9)
            x_noise, y_noise = make_noise(make_borrow=make_borrow)

        elif test_case == TriCaseBehavior.MT2:
            # These are n_digit subtraction questions where x - y is 0
            x = random.randint(0, 9)
            y = x
            if qtype == QType.MATH_NEG:
                x_noise, y_noise = make_noise(make_borrow="always")
            else:
                x_noise, y_noise = make_noise(make_borrow="never")

        elif test_case == TriCaseBehavior.MT1:
            # These are n_digit subtraction questions where x - y > 0
            x = random.randint(1, 9)
            y = random.randint(0, x - 1)
            x_noise, y_noise = make_noise(make_borrow)

        else:
            raise Exception(f'Only behaviors MT1, MT2, and MT3 supported for negative. Received {test_case}.')

    else:
        raise Exception(f'Only MINUS and PLUS operations are supported currently, received {operation}.')

    x = x * limit + x_noise
    y = y * limit + y_noise

    if x -y < 0 and qtype == QType.MATH_SUB:
        raise Exception("Math_sub should have positive result")
    if x -y >= 0 and qtype == QType.MATH_NEG:
        raise Exception("Math_neg should have negative result")

    return x, y

def make_tricase_questions(
        cfg, test_digit: int, test_case: TriCaseBehavior, operation: MathsToken, num_questions=EACH_CASE_TRICASE_QUESTIONS, qtype: QType = None,
        make_borrow: str = "never", make_carry: str = "never"
):
    """
    Returns a set of questions over a number of test digits, given an operation and optionally a qtype
    make_borrow should be "never", "always" or "mixed" and controls whether we allow for
    borrow ones in less significant digits or not.
    make_borrow will be overidden if the case_type and qtype require it, such as for
    tricase test_case=8 and qtype=Math_NEG
    """
    print(f'Making {num_questions} questions for {test_digit} and {test_case} with qtype {qtype}.')
    assert make_borrow in ["always", "never", "mixed"]
    assert make_carry in ["always", "never", "mixed"]
    questions = []
    exceptions = []
    assert qtype in [None, QType.MATH_ADD, QType.MATH_SUB, QType.MATH_NEG, QType.UNKNOWN], f"Qtype must be none, sub, neg or unknown, but received {qtype}"
    assert isinstance(test_case, TriCaseBehavior), f"Tricase test cases must be of type TriCaseBehavior"
    assert operation in [MathsToken.PLUS, MathsToken.MINUS], f"Tricase operation must be in [plus,minus]={[MathsToken.PLUS, MathsToken.MINUS]}, received operation {operation}"

    attempts = 0
    # Attempts stops us from trying forever if the requested operation is impossible.
    while len(set(questions)) < num_questions and attempts <= 20*num_questions:
        attempts +=1
        try:
            x,y = make_single_tricase_question(
                cfg=cfg, test_digit=test_digit, test_case=test_case, operation=operation,
                qtype=qtype, make_borrow=make_borrow)
            questions.append((x,y))

        except Exception as e:
            print(f'Caught exception {e} on test case {test_case} on digit {test_digit} and qtype {qtype}.')
            exceptions.append(e)

    questions = list(set(questions))


    if len(exceptions):
        print(f'Received {len(exceptions)} exceptions creating {len(questions)} questions out of {num_questions} for test case {test_case} on digit {test_digit} and qtype {qtype}.')

    if len(questions) < num_questions and test_digit<3:
        questions = pad_small_set_of_questions(
            cfg, sample_pairs_of_numbers=questions, target_number=num_questions, digit=test_digit
        )

    print(f'Generated {len(questions)} questions for operation {operation}, which include:\n{questions[:3]}.')

    if qtype is not None:  # We have enforced qtype remains consistent with questions returned
        result = make_maths_questions_and_answers(cfg, operation, qtype, MathsBehavior.UNKNOWN, questions)
        return result

    elif operation == MathsToken.PLUS:  # qtype not relevant for MathsToken.PLUS
        qtype = QType.MATH_ADD #if operation == MathsToken.PLUS else QType.MATH_SUB # Inaccurate. Will be a mix of QType.MATH_SUB and QType.MATH_NEG

        result = make_maths_questions_and_answers(cfg, operation, qtype, MathsBehavior.UNKNOWN, questions)
        print(f'returned result of length {result}')
        return result

    elif operation == MathsToken.MINUS:
        sub_questions = [question for question in questions if question[0] >= question[1]]
        sub_question_tensors = make_maths_questions_and_answers(cfg, operation, QType.MATH_SUB, MathsBehavior.UNKNOWN, sub_questions)

        neg_questions = [question for question in questions if question[0] < question[1]]
        neg_question_tensors = make_maths_questions_and_answers(cfg, operation, QType.MATH_NEG, MathsBehavior.UNKNOWN, neg_questions)

        return torch.vstack([sub_question_tensors, neg_question_tensors])

    else:
        raise Exception(f"Unsupported operation {operation}.")

def make_maths_tricase_questions_core(cfg, test_digit, operation, num_questions=TOTAL_TRICASE_QUESTIONS):
    assert num_questions%3==0, "Number of questions must be divisible by 3"
    local_num_questions = int(num_questions/3)

    if operation == MathsToken.PLUS:
        q1 = make_tricase_questions(cfg, test_digit, TriCaseBehavior.ST8, operation, num_questions=local_num_questions)
        q2 = make_tricase_questions(cfg, test_digit, TriCaseBehavior.ST8, operation, num_questions=local_num_questions)
        q3 = make_tricase_questions(cfg, test_digit, TriCaseBehavior.ST10, operation, num_questions=local_num_questions)

    elif operation == MathsToken.MINUS:
        q1 = make_tricase_questions(cfg, test_digit, TriCaseBehavior.MT1, operation, num_questions=local_num_questions)
        q2 = make_tricase_questions(cfg, test_digit, TriCaseBehavior.MT2, operation, num_questions=local_num_questions)
        q3 = make_tricase_questions(cfg, test_digit, TriCaseBehavior.MT3, operation, num_questions=local_num_questions)

    else:
        raise Exception(f'Only PLUS and MINUS operations are currently supported, received {operation}')
    return torch.vstack((q1, q2, q3))

def make_maths_tricase_questions(cfg, num_questions=TOTAL_TRICASE_QUESTIONS):
    cfg.tricase_questions_dict = {}
    for answer_digit in range(cfg.n_digits):
        for operation in [MathsToken.PLUS, MathsToken.MINUS]:
            t_questions = make_maths_tricase_questions_core(cfg, answer_digit, operation, num_questions=num_questions)
            # Use a tuple of (answer_digit, operation) as the key for indexing
            cfg.tricase_questions_dict[(answer_digit, operation)] = t_questions


def make_maths_tricase_questions_customized(cfg, custom_triclass_config=CustomTriclassConfig(), verbose=False):
    """
    Creates a dictionary of tricase questions in cfg.tricase_questions_dict.
    This dictionary is indexed by (answer_digit, operator, question_type) with custom_triclass_config.number examples of each.
    """
    cfg.customized_tricase_questions_dict = {}
    for answer_digit in range(cfg.n_digits):
        operators_qtype_numbers = custom_triclass_config.operators_qtypes_counts

        for operator_qtype_number in operators_qtype_numbers:
            num_questions = operator_qtype_number.number
            qtype = operator_qtype_number.qtype
            operator = operator_qtype_number.operator

            if qtype in [QType.MATH_NEG, QType.MATH_SUB] and operator == MathsToken.PLUS:
                raise Exception(f'A qtype of MATH_NEG or MATH_SUB is not supported with plus operator.')

            elif qtype == QType.MATH_SUB:
                # Only test cases MT1 and MT2 are supported for MATH_SUB
                target_cases = [TriCaseBehavior.MT1, TriCaseBehavior.MT2]
                local_num_questions = int(num_questions / len(target_cases))
                for test_case in target_cases:
                    all_questions = make_tricase_questions(
                        cfg, test_digit=answer_digit, test_case=test_case, operation=operator, qtype=qtype, num_questions=local_num_questions
                    )
                    key = DigitOperatorQTypeTricase(answer_digit, maths_tokens_to_names[operator], qtype, test_case)
                    cfg.customized_tricase_questions_dict[key] = all_questions

            elif qtype == QType.MATH_NEG:
                # Only test cases MT2 and MT3 are supported for MATH_NEG when digit > 0, and only test case MT3 for digit=0
                target_cases = [TriCaseBehavior.MT2, TriCaseBehavior.MT3] if answer_digit > 0 else [TriCaseBehavior.MT3]
                local_num_questions = int(num_questions / len(target_cases))
                for test_case in target_cases:
                    all_questions = make_tricase_questions(
                        cfg, test_digit=answer_digit, test_case=test_case, operation=operator, qtype=qtype,
                        num_questions=local_num_questions
                    )
                    key = DigitOperatorQTypeTricase(answer_digit, maths_tokens_to_names[operator], qtype, test_case)
                    cfg.customized_tricase_questions_dict[key] = all_questions


            elif qtype == QType.MATH_ADD:
                target_cases = [TriCaseBehavior.ST8, TriCaseBehavior.ST9, TriCaseBehavior.ST10]
                local_num_questions = int(num_questions / len(target_cases))
                for test_case in target_cases:
                    all_questions = make_tricase_questions(
                        cfg, test_digit=answer_digit, test_case=test_case, operation=operator, qtype=qtype,
                        num_questions=local_num_questions
                    )
                    print(f'Received back {len(all_questions)} for test case {test_case.name} and operator {operator}.')
                    key = DigitOperatorQTypeTricase(answer_digit, maths_tokens_to_names[operator], qtype, test_case)
                    cfg.customized_tricase_questions_dict[key] = all_questions

            else:
                raise Exception(f'Unknown qtype {qtype}.')

            questions_created = [len(cfg.customized_tricase_questions_dict.get(
                DigitOperatorQTypeTricase(answer_digit, maths_tokens_to_names[operator], qtype, test_case), [])) for test_case in TriCaseBehavior
            ]
            num_questions_created = sum(questions_created)
            assert num_questions_created == num_questions, (
                f"Created {num_questions_created}  for digit {answer_digit} "
                f"when requested with {num_questions} for {operator_qtype_number}")

    value_distribution = {key: len(values) for key, values in cfg.customized_tricase_questions_dict.items()}

    if verbose:
        print(f'Value distribution for (answer_digit, operator, qtype) is: \n{value_distribution}')

    return cfg.customized_tricase_questions_dict.copy()
