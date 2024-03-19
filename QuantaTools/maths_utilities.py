from .maths_vocab import MathsTokens


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
    sign_offset = cfg.question_tokens()

    # 5 digit addition yields a 6 digit answer. So cfg.n_digits+1 DIGITS
    answer_digits = cfg.n_digits+1

    a = tokens_to_unsigned_int( q, sign_offset+1, answer_digits )
    if q[sign_offset] == MathsTokens.MINUS:
        a = - a

    return a


# Insert a number into the question
def insert_question_number(the_question, index, first_digit_index, the_digits, n):

    last_digit_index = first_digit_index + the_digits - 1

    for j in range(the_digits):
        the_question[index, last_digit_index-j] = n % 10
        n = n // 10


# Create a single maths question and answer
def make_a_maths_question_and_answer(cfg, the_question, index, q1, q2, operator ):

    insert_question_number(the_question, index, 0, cfg.n_digits, q1)

    the_question[index, cfg.n_digits] = operator

    insert_question_number( the_question, index, cfg.n_digits+1, cfg.n_digits, q2)

    the_question[index, 2*cfg.n_digits+1] = MathsTokens.EQUALS

    answer = q1+q2
    if operator == MathsTokens.MINUS:
        answer = q1-q2
    elif operator == MathsTokens.MULT:
        answer = q1*q2

    the_question[index, cfg.question_tokens()] = MathsTokens.PLUS if answer >= 0 else MathsTokens.MINUS
    if answer < 0:
        answer = -answer

    insert_question_number(the_question, index, 2*cfg.n_digits + 3, cfg.n_digits+1, answer)
