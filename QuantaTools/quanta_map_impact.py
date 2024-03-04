from .quanta_filter import QuantaFilter
from .quanta_type import QuantaType
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, NodeLocation, UsefulNode 
from .useful_info import UsefulInfo, useful_info
from .token_to_char import token_to_char, tokens_to_string 


# Compare each digit in the answer. Returns a A645 pattern where '4' means a failed 4th digit. A "-" sign failure is shown as "A7"
def get_answer_impact_meaning_str(answer1_str, answer2_str):

  impact = ""
  sign_offset = useful_info.question_tokens()
  for i in range(useful_info.answer_tokens()):
    impact += "" if answer2_str[i] == answer1_str[i] else useful_info.token_position_meanings[sign_offset + i]

  if impact == "":
    return ""

  impact = impact.replace("A", "")
  char_list = list(impact)
  char_list = sorted(char_list, reverse = not useful_info.answer_meanings_ascend)
  impact = ''.join(char_list)

  return "A" + impact


# Compare each digit in the answer. Returns a A645 pattern where '4' means a failed 4th answer digit.
def get_answer_impact_meaning(question_and_answer, answer_str2):

  answer1_str = tokens_to_string(question_and_answer[-useful_info.num_answer_positions:])

  return get_answer_impact_meaning_str(answer1_str, answer_str2)


# Check if the digits in the string are sequential e.g. A1234 or A4321
def is_answer_sequential(digits):
  if useful_info.answer_meanings_ascend :
    return all(ord(next_char) - ord(current_char) == 1 for current_char, next_char in zip(digits, digits[1:]))
  else:
    return all(ord(next_char) - ord(current_char) == -1 for current_char, next_char in zip(digits, digits[1:]))


# Convert A654321 to A6..1, or A123456 to A1..6 for compact display
def compact_answer_if_sequential(s):
    if len(s) > 3:
      letter, digits = s[0], s[1:]
      if is_answer_sequential(digits, useful_info.answer_meanings_ascend):
        # Convert to compact form 
        if useful_info.answer_meanings_ascend:
          return f"{letter}{digits[-1]}..{digits[0]}"
        else:
          return f"{letter}{digits[0]}..{digits[-1]}"

    # Return original string if not sequential
    return s
