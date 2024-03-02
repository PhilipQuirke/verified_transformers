from .quanta_filter import QuantaFilter
from .quanta_type import QuantaType
from .useful_node import position_name, position_name_to_int, row_location_name, location_name, NodeLocation, UsefulNode 
from .useful_info import UsefulInfo, useful_info


# convert 3 to "A3"
def answer_name(n):
  return "A" + str(n)


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

  answer1_str = tokens_to_string(question_and_answer[-useful_info.answer_tokens():])

  return get_answer_impact_meaning_str(answer1_str, answer_str2)
