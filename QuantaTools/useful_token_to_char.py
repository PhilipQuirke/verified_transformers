from .useful_info import UsefulInfo


# Map from token to character
def token_to_char(useful_info, i):
  for char, token in useful_info.char_to_token.items():
    if token == i:
      return char

  # Should never happen
  assert False


# Map from tokens to string
def tokens_to_string(tokens):
    return "".join([token_to_char(i) for i in tokens])
