import unittest

from QuantaTools.maths_config import MathsConfig
from QuantaTools.maths_utilities import int_to_answer_str, tokens_to_unsigned_int




class TestYourModule(unittest.TestCase):
    def test_int_to_answer_str(self):
        cfg = MathsConfig()
        cfg.n_digits = 6
        self.assertEqual( int_to_answer_str(1234), "+0001234" )

        
    def test_tokens_to_unsigned_int(self):
        q = [0,0,1,2,3,4,5]
        offset = 0
        digits = 6
        self.assertEqual( tokens_to_unsigned_int(q, offset, digits), 12345 )


if __name__ == '__main__':
    unittest.main()
