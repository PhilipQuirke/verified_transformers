from ..\QuantaTools\useful_info import useful_info


# Vocabulary dictionary: Mapping from character (key) to token (value)
def set_maths_vocabulary():
  useful_info.char_to_token = {str(i) : i for i in range(10)}
  useful_info.char_to_token['+'] = PLUS_INDEX
  useful_info.char_to_token['-'] = MINUS_INDEX
  useful_info.char_to_token['='] = EQUALS_INDEX
  useful_info.char_to_token['*'] = MULT_INDEX
  useful_info.char_to_token['\\'] = DIV_INDEX
