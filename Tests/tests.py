import unittest

from QuantaTools.model_token_to_char import token_to_char, tokens_to_string

from QuantaTools.quanta_map_impact import sort_unique_digits

from QuantaTools.maths_config import MathsConfig
from QuantaTools.maths_vocab import MathsTokens
from QuantaTools.maths_vocab import set_maths_vocabulary
from QuantaTools.maths_utilities import int_to_answer_str, tokens_to_unsigned_int


class TestYourModule(unittest.TestCase):


    def test_int_to_answer_str(self):
        cfg = MathsConfig()
        cfg.n_digits = 6
        self.assertEqual( int_to_answer_str(cfg, 1234), "+0001234" )

        
    def test_tokens_to_unsigned_int(self):
        q = [0,0,1,2,3,4,5]
        offset = 0
        digits = 6
        self.assertEqual( tokens_to_unsigned_int(q, offset, digits), 12345 )


    def test_set_maths_vocabulary(self):
        cfg = MathsConfig()
        set_maths_vocabulary(cfg)

        self.assertEqual( token_to_char(cfg, 4), '4')
        self.assertEqual( token_to_char(cfg, MathsTokens.MULT), '*')
        self.assertEqual( tokens_to_string(cfg, [MathsTokens.EQUALS,4,0,7]), '=407')
  

    def test_sort_unique_digits(self):
        self.assertEqual( sort_unique_digits("A1231231278321", False), "12378")
        self.assertEqual( sort_unique_digits("A1231231278321", True), "87321")



if __name__ == '__main__':
    unittest.main()
