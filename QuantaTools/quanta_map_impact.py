from .model_token_to_char import tokens_to_string 


# Compare the digits in say "+0017726" and "+0018826" returning "A32" where '2' means a mismatch in digit A2. A "-" sign failure is shown as say "A7"
def get_answer_impact(cfg, answer1_str, answer2_str):
    assert len(answer1_str) >= cfg.num_answer_positions
    assert len(answer2_str) >= cfg.num_answer_positions

    impact = ""
    sign_offset = cfg.num_question_positions 
    for i in range(cfg.num_answer_positions):
        impact += "" if answer2_str[i] == answer1_str[i] else cfg.token_position_meanings[sign_offset + i]

    if impact == "":
        return ""

    impact = impact.replace("A", "")
    char_list = list(impact)
    char_list = sorted(char_list, reverse = not cfg.answer_meanings_ascend)
    impact = ''.join(char_list)

    return "A" + impact


# Compare each digit in the answer. Returns a A645 pattern where '4' means a failed 4th answer digit.
def get_question_answer_impact(cfg, question_and_answer, answer_str2):

    answer1_str = tokens_to_string(cfg, question_and_answer[-cfg.num_answer_positions:])

    return get_answer_impact(cfg, answer1_str, answer_str2)


# Check if the digits in the string are sequential e.g. A1234 or A4321
def is_answer_sequential(cfg, digits):
    if cfg.answer_meanings_ascend :
        return all(ord(next_char) - ord(current_char) == 1 for current_char, next_char in zip(digits, digits[1:]))
    else:
        return all(ord(next_char) - ord(current_char) == -1 for current_char, next_char in zip(digits, digits[1:]))


# Convert A654321 to A6..1, or A123456 to A1..6 for compact display
def compact_answer_if_sequential(cfg, s):
    if len(s) > 3:
        letter, digits = s[0], s[1:]
        if is_answer_sequential(cfg, digits):
            # Convert to compact form 
            return f"{letter}{digits[0]}..{digits[-1]}"

    # Return original string if not sequential
    return s


def get_quanta_impact( cfg, node, major_tag : str, minor_tag : str, num_shades : int ):

    cell_text = ""
    color_index = 0

    cell_text = node.min_tag_suffix( major_tag, minor_tag )
    if len(cell_text) > 0:
        cell_text = compact_answer_if_sequential(cfg, cell_text)

        color_index = int(cell_text[1]) if len(cell_text) > 1 and cell_text[1].isdigit() else num_shades-1

    return cell_text, color_index


# Convert "A1231231278321" to "12378" or "87321"
def sort_unique_digits(raw_input_string, do_reverse):
    digit_string = ''.join(filter(str.isdigit, raw_input_string))

    seen = set()
    unique_digits = ""
    for char in digit_string:
        if char not in seen:
            seen.add(char)
            unique_digits += char

    return ''.join(sorted(unique_digits, reverse=do_reverse))
