
class MathsTokens:
    # Tokens used in arithmetic vocab. (Token indexes 0 to 9 represent digits 0 to 9)
    PLUS = 10
    MINUS = 11
    EQUALS = 12
    MULT = 13
    DIV = 14
    MAX_INDEX = DIV


# Vocabulary dictionary: Mapping from character (key) to token (value)
def set_maths_vocabulary(cfg):
    cfg.char_to_token = {str(i) : i for i in range(10)}
    cfg.char_to_token['+'] = MathsTokens.PLUS
    cfg.char_to_token['-'] = MathsTokens.MINUS
    cfg.char_to_token['='] = MathsTokens.EQUALS
    cfg.char_to_token['*'] = MathsTokens.MULT
    cfg.char_to_token['\\'] = MathsTokens.DIV


# Maths question and answer token position meanings are D5, .., D0, *, D5', .., D0', =, A7, A6, .., A0
def set_maths_question_meanings(cfg):
    q_meanings = []
    for i in range(cfg.n_digits):
        q_meanings += ["D" + str(cfg.n_digits-i-1)]
    q_meanings += ["Op"] # Stands in for operation +, - or *
    for i in range(cfg.n_digits):
        q_meanings += ["D'" + str(cfg.n_digits-i-1)]
    q_meanings += ["="]

    cfg.token_position_meanings = q_meanings + cfg.token_position_meanings[-cfg.num_answer_positions:]
