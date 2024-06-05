from .maths_constants import MathsToken


# Vocabulary dictionary: Mapping from character (key) to token (value)
def set_maths_vocabulary(cfg):
    cfg.char_to_token = {str(i) : i for i in range(10)}
    cfg.char_to_token['+'] = MathsToken.PLUS
    cfg.char_to_token['-'] = MathsToken.MINUS
    cfg.char_to_token['='] = MathsToken.EQUALS
    cfg.char_to_token['*'] = MathsToken.MULT
    cfg.char_to_token['\\'] = MathsToken.DIV


def digit_name(digit):
    return "D" + str(digit)


# Maths question and answer token position meanings are D5, .., D0, *, D5', .., D0', =, A7, A6, .., A0
def set_maths_question_meanings(cfg):
    q_meanings = []
    for i in range(cfg.n_digits):
        q_meanings += [digit_name(cfg.n_digits-i-1)]
    q_meanings += ["OPR"] # Stands in for operation +, - or *
    for i in range(cfg.n_digits):
        q_meanings += ["D'" + str(cfg.n_digits-i-1)]
    q_meanings += ["="]

    cfg.token_position_meanings = q_meanings + cfg.token_position_meanings[-cfg.num_answer_positions:]


def int_to_answer_str( cfg, n ):
    s = str(abs(n))
    while len(s) < cfg.n_digits + 1 :
        s = "0" + s
    s = ("+" if n >= 0 else "-") + s
    return s


# Convert "0012345" to 12345
def tokens_to_unsigned_int( q, offset, digits ):
    a = 0
    for j in range(digits):
        a = a * 10 + q[offset+j]
    return a


# Convert "-12345" to -12345, and "+12345" to 12345
def tokens_to_answer(cfg, q):
    # offset of sign character
    sign_offset = cfg.num_question_positions

    # 5 digit addition yields a 6 digit answer. So cfg.n_digits+1 DIGITS
    answer_digits = cfg.n_digits+1

    a = tokens_to_unsigned_int( q, sign_offset+1, answer_digits )
    if q[sign_offset] == MathsToken.MINUS:
        a = - a

    return a


# Insert a number into the question
def insert_question_number(the_question, index, first_digit_index, the_digits, n):

    last_digit_index = first_digit_index + the_digits - 1

    for j in range(the_digits):
        the_question[index, last_digit_index-j] = n % 10
        n = n // 10


# Create a single maths question and answer, by writing to 2d matrix the_question.
def make_a_maths_question_and_answer(cfg, the_question, index, q1, q2, operator ):
    insert_question_number(the_question, index, 0, cfg.n_digits, q1)

    the_question[index, cfg.n_digits] = operator

    insert_question_number( the_question, index, cfg.n_digits+1, cfg.n_digits, q2)

    the_question[index, 2*cfg.n_digits+1] = MathsToken.EQUALS

    answer = q1+q2
    if operator == MathsToken.MINUS:
        answer = q1-q2
    elif operator == MathsToken.MULT:
        answer = q1*q2

    the_question[index, cfg.num_question_positions] = MathsToken.PLUS if answer >= 0 else MathsToken.MINUS
    if answer < 0:
        answer = -answer

    insert_question_number(the_question, index, 2*cfg.n_digits + 3, cfg.n_digits+1, answer)
