import random
from typing import Optional

import torch

from QuantaTools import MathsTask, MathsToken, QType, make_maths_questions_and_answers, MathsBehavior


# Create a cache of sample (matrix) maths questions based on the T8, T9, T10 categorisation
DEFAULT_TRICASE_QUESTIONS = 100


def make_tricase_questions(
        cfg, test_digit: int, test_case: int, operation: MathsToken,
        requested_features: Optional[list[MathsTask]] =None, excluded_features: Optional[list[MathsTask]] = None,
        num_tricase_questions: int = DEFAULT_TRICASE_QUESTIONS
):
    """
    Returns a set of questions over a number of test digits, given an operation,
    requested features and excluded features corresponding to MathsTasks.
    """

    limit = 10 ** test_digit
    questions = []
    for i in range(num_tricase_questions):
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


def make_maths_tricase_questions(cfg):
    cfg.tricase_questions_dict = {}
    for answer_digit in range(cfg.n_digits):
        for operation in [MathsToken.PLUS, MathsToken.MINUS]:
            t_questions = make_maths_tricase_questions_core(cfg, answer_digit, operation)
            # Use a tuple of (answer_digit, operation) as the key for indexing
            cfg.tricase_questions_dict[(answer_digit, operation)] = t_questions
