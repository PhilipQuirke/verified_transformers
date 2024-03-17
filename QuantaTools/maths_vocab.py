from .model_token_to_char import token_to_char, tokens_to_string


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

  # Unit tests
  assert token_to_char(cfg, 4) == '4'
  assert token_to_char(cfg, MathsTokens.MULT) == '*'
  assert tokens_to_string(cfg, [MathsTokens.EQUALS,4,0,7]) == '=407'
