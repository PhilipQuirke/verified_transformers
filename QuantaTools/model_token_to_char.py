

# Map from token to character
def token_to_char(cfg, i):
    for char, token in cfg.char_to_token.items():
        if token == i:
            return char

    # Should never happen
    assert False


# Map from tokens to string
def tokens_to_string(cfg,tokens):
    return "".join([token_to_char(cfg,i) for i in tokens])
