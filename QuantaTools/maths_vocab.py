from .useful_token_to_char import token_to_char, tokens_to_string


class MathsTokens:
  # Tokens used in arithmetic vocab. (Token indexes 0 to 9 represent digits 0 to 9)
  PLUS = 10
  MINUS = 11
  EQUALS = 12
  MULT = 13
  DIV = 14
  MAX_INDEX = DIV


# Vocabulary dictionary: Mapping from character (key) to token (value)
def set_maths_vocabulary(useful_info):
  useful_info.char_to_token = {str(i) : i for i in range(10)}
  useful_info.char_to_token['+'] = MathsTokens.PLUS
  useful_info.char_to_token['-'] = MathsTokens.MINUS
  useful_info.char_to_token['='] = MathsTokens.EQUALS
  useful_info.char_to_token['*'] = MathsTokens.MULT
  useful_info.char_to_token['\\'] = MathsTokens.DIV

  # Unit tests
  assert token_to_char(4) == '4'
  assert token_to_char(MathsTokens.MULT_INDEX) == '*'
  assert tokens_to_string([MathsTokens.EQUALS_INDEX,4,0,7]) == '=407'
