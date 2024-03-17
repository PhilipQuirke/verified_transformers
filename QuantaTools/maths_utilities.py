from .maths_vocab import MathsTokens
from .model_config import ModelConfig


def int_to_answer_str( cfg, n ):
    s = str(abs(n))
    while len(s) < cfg.n_digits + 1 :
        s = "0" + s
    s = ("+" if n >= 0 else "-") + s
    return s


# Unit test
# if cfg.n_digits == 6 :
#    assert int_to_answer_str(1234) == "+0001234"


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