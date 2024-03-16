from .useful_info import useful_info


class MathsTokens:
  # Tokens used in arithmetic vocab. (Token indexes 0 to 9 represent digits 0 to 9)
  PLUS_INDEX = 10
  MINUS_INDEX = 11
  EQUALS_INDEX = 12
  MULT_INDEX = 13
  DIV_INDEX = 14
  MAX_INDEX = DIV_INDEX


# Vocabulary dictionary: Mapping from character (key) to token (value)
def set_maths_vocabulary():
  useful_info.char_to_token = {str(i) : i for i in range(10)}
  useful_info.char_to_token['+'] = PLUS_INDEX
  useful_info.char_to_token['-'] = MINUS_INDEX
  useful_info.char_to_token['='] = EQUALS_INDEX
  useful_info.char_to_token['*'] = MULT_INDEX
  useful_info.char_to_token['\\'] = DIV_INDEX
