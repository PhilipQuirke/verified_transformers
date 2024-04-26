import unittest

from QuantaTools.model_token_to_char import token_to_char, tokens_to_string

from QuantaTools.useful_node import NodeLocation, UsefulNode, UsefulNodeList

from QuantaTools.quanta_constants import QType, MATH_ADD_SHADES, MATH_SUB_SHADES
from QuantaTools.quanta_map_impact import sort_unique_digits, get_quanta_impact

from QuantaTools.maths_tools.maths_config import MathsConfig
from QuantaTools.maths_tools.maths_constants import MathsToken, MathsBehavior
from QuantaTools.maths_tools.maths_utilities import set_maths_vocabulary, int_to_answer_str, tokens_to_unsigned_int
from QuantaTools.maths_tools.maths_test_questions import make_maths_test_questions_and_answers, make_maths_questions_and_answers
from QuantaTools.maths_tools.maths_test_questions import make_maths_s0_questions_and_answers, make_maths_s1_questions_and_answers, make_maths_s2_questions_and_answers, make_maths_s3_questions_and_answers, make_maths_s4_questions_and_answers, make_maths_s5_questions_and_answers
from QuantaTools.maths_tools.maths_test_questions import make_maths_m0_questions_and_answers, make_maths_m1_questions_and_answers, make_maths_m2_questions_and_answers, make_maths_m3_questions_and_answers
from QuantaTools.maths_tools.maths_test_questions import make_maths_n1_questions_and_answers, make_maths_n2_questions_and_answers, make_maths_n3_questions_and_answers, make_maths_n4_questions_and_answers
from QuantaTools.maths_tools.maths_complexity import get_maths_min_complexity


class TestUseful(unittest.TestCase):


    def test_useful_node(self):
        
        useful_node = UsefulNode(1, 2, True, 3, [])

        # Add 6 distinct tags
        useful_node.add_tag( "Major0", "Minor1" )
        useful_node.add_tag( "Major0", "Minor2" )
        useful_node.add_tag( "Major1", "Minor1" )
        useful_node.add_tag( "Major1", "Minor2" )
        useful_node.add_tag( "Major2", "Minor1" )
        useful_node.add_tag( "Major2", "Minor2" )
        self.assertEqual( len(useful_node.tags), 6)
        
        # Add 2 duplicate tags
        useful_node.add_tag( "Major1", "Minor1" )
        useful_node.add_tag( "Major2", "Minor2" )
        self.assertEqual( len(useful_node.tags), 6)

        # Remove 2 tags
        useful_node.reset_tags( "Major2" )
        self.assertEqual( len(useful_node.tags), 4)

        # Remove 1 tag
        useful_node.reset_tags( "Major0", "Minor2" )
        self.assertEqual( len(useful_node.tags), 3)

        # Remove all tags
        useful_node.reset_tags( "" )
        self.assertEqual( len(useful_node.tags), 0)


class TestMaths(unittest.TestCase):


    def test_int_to_answer_str(self):
        cfg = MathsConfig()
        cfg.n_digits = 6
        self.assertEqual( int_to_answer_str(cfg, 1234), "+0001234" )

        
    def test_tokens_to_unsigned_int(self):
        q = [0,1,2,3,4,5]
        offset = 0
        digits = 6
        self.assertEqual( tokens_to_unsigned_int(q, offset, digits), 12345 )


    def test_set_maths_vocabulary(self):
        cfg = MathsConfig()
        set_maths_vocabulary(cfg)

        self.assertEqual( token_to_char(cfg, 4), '4')
        self.assertEqual( token_to_char(cfg, MathsToken.MULT), '*')
        self.assertEqual( tokens_to_string(cfg, [MathsToken.EQUALS,4,0,7]), '=407')
  

    def test_sort_unique_digits(self):
        self.assertEqual( sort_unique_digits("A1231231278321", False), "12378")
        self.assertEqual( sort_unique_digits("A1231231278321", True), "87321")


    # During the construction of the test data, we check that the complexity of the question matches what the test data believes it is
    def test_make_maths_test_questions_and_answers(self):
        cfg = MathsConfig()
        cfg.perc_sub = 60
        cfg.use_cuda = False
        set_maths_vocabulary(cfg)
        
        # Addition questions 
        make_maths_s0_questions_and_answers(cfg)
        make_maths_s1_questions_and_answers(cfg)
        make_maths_s2_questions_and_answers(cfg)
        make_maths_s3_questions_and_answers(cfg)
        make_maths_s4_questions_and_answers(cfg)
        make_maths_s5_questions_and_answers(cfg)
            
        # Subtraction questions with positive (or zero) answers
        make_maths_m0_questions_and_answers(cfg) 
        make_maths_m1_questions_and_answers(cfg)
        make_maths_m2_questions_and_answers(cfg)
        make_maths_m3_questions_and_answers(cfg)
            
        # Subtraction questions with negative answers
        make_maths_n1_questions_and_answers(cfg)         
        make_maths_n2_questions_and_answers(cfg)
        make_maths_n3_questions_and_answers(cfg)
        make_maths_n4_questions_and_answers(cfg)
      
  
    def test_repeat_digit(self):
        cfg = MathsConfig()
        cfg.n_digits = 6
        
        self.assertEqual( cfg.repeat_digit(4), 444444)



    def test_useful_node_list(self):
        cfg = MathsConfig()
        cfg.perc_sub = 60
        cfg.use_cuda = False
        set_maths_vocabulary(cfg)
  
        list = UsefulNodeList()

        # Data from ins1_mix_d6_l3_h4_t40K_s372001 for P18L0
        
        location = NodeLocation(18,0,True,0)
        list.add_node_tag( location, QType.FAIL.value, '6' )
        list.add_node_tag( location, QType.IMPACT.value, 'A2' )
        list.add_node_tag( location, QType.MATH_ADD.value, 'S123' )
        list.add_node_tag( location, QType.MATH_SUB.value, 'M123' )
        list.add_node_tag( location, QType.ATTN.value, 'P4=50' )
        list.add_node_tag( location, QType.ATTN.value, 'P11=50' )
        list.add_node_tag( location, QType.MATH_ADD.value, 'A0.SP.Weak' ) 
        list.add_node_tag( location, QType.MATH_ADD.value, 'A2.SP.Weak' )      
        list.add_node_tag( location, QType.MATH_SUB.value, 'A0.MP.Weak' )      
        list.add_node_tag( location, QType.MATH_SUB.value, 'A2.MP.Weak' )      
     
        location = NodeLocation(18,0,True,1)
        list.add_node_tag( location, QType.FAIL.value, '44' )
        list.add_node_tag( location, QType.IMPACT.value, 'A2' )
        list.add_node_tag( location, QType.MATH_ADD.value, 'S01234' )
        list.add_node_tag( location, QType.MATH_SUB.value, 'M0123' )
        list.add_node_tag( location, QType.MATH_NEG.value, 'N1234' )
        list.add_node_tag( location, QType.ATTN.value, 'P3=52' )
        list.add_node_tag( location, QType.ATTN.value, 'P10=48' )
        list.add_node_tag( location, QType.MATH_ADD.value, 'A0.SP.Weak' ) 
        list.add_node_tag( location, QType.MATH_ADD.value, 'A1.SP.Weak' )      
        list.add_node_tag( location, QType.MATH_SUB.value, 'A0.MP.Weak' )      
        list.add_node_tag( location, QType.MATH_SUB.value, 'A1.MP.Weak' )   

        location = NodeLocation(18,0,True,2)
        list.add_node_tag( location, QType.FAIL.value, '58' )
        list.add_node_tag( location, QType.IMPACT.value, 'A2' )
        list.add_node_tag( location, QType.MATH_ADD.value, 'S012345' )
        list.add_node_tag( location, QType.MATH_SUB.value, 'M0123' )
        list.add_node_tag( location, QType.MATH_NEG.value, 'N1234' )
        list.add_node_tag( location, QType.ATTN.value, 'P10=51' )
        list.add_node_tag( location, QType.ATTN.value, 'P3=47' )

        location = NodeLocation(18,0,True,3)
        list.add_node_tag( location, QType.FAIL.value, '12' )
        list.add_node_tag( location, QType.IMPACT.value, 'A2' )
        list.add_node_tag( location, QType.MATH_SUB.value, 'M0123' )
        list.add_node_tag( location, QType.MATH_NEG.value, 'N1234' )
        list.add_node_tag( location, QType.ATTN.value, 'P6=86' )
        list.add_node_tag( location, QType.ATTN.value, 'P14=6' )
        list.add_node_tag( location, QType.ATTN.value, 'P13=3' )
        list.add_node_tag( location, QType.ATTN.value, 'P11=2' )
        list.add_node_tag( location, QType.MATH_ADD.value, 'A2.SP' ) 
        list.add_node_tag( location, QType.MATH_SUB.value, 'A0.MP.Weak' )      
        list.add_node_tag( location, QType.MATH_SUB.value, 'A1.MP.Weak' )   
        list.add_node_tag( location, QType.MATH_SUB.value, 'A2.MP.Weak' )      
        list.add_node_tag( location, QType.MATH_SUB.value, 'A3.MP.Weak' )         
        list.add_node_tag( location, QType.MATH_SUB.value, 'A4.MP.Weak' )      
        list.add_node_tag( location, QType.MATH_SUB.value, 'A5.MP.Weak' )      
  
        location = NodeLocation(18,0,False,0)
        list.add_node_tag( location, QType.FAIL.value, '50' )
        list.add_node_tag( location, QType.IMPACT.value, 'A2' )
        list.add_node_tag( location, QType.MATH_ADD.value, 'S012345' )         
        list.add_node_tag( location, QType.MATH_SUB.value, 'M0123' )
        list.add_node_tag( location, QType.MATH_NEG.value, 'N1234' )

        location = NodeLocation(18,0,True,0)
        node = list.get_node( location )
        node_add_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "S1")
        self.assertEqual( node_sub_complexity, "M1")
        self.assertEqual( node_neg_complexity, "")

        location = NodeLocation(18,0,True,1)
        node = list.get_node( location )        
        node_add_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "S0")
        self.assertEqual( node_sub_complexity, "M0")
        self.assertEqual( node_neg_complexity, "N1")

        location = NodeLocation(18,0,True,2)
        node = list.get_node( location )        
        node_add_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "S0")
        self.assertEqual( node_sub_complexity, "M0")
        self.assertEqual( node_neg_complexity, "N1")

        location = NodeLocation(18,0,True,3)
        node = list.get_node( location )        
        node_add_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "")
        self.assertEqual( node_sub_complexity, "M0")
        self.assertEqual( node_neg_complexity, "N1")
        
        location = NodeLocation(18,0,False,0)
        node = list.get_node( location )        
        node_add_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_maths_min_complexity( cfg, node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "S0")
        self.assertEqual( node_sub_complexity, "M0")
        self.assertEqual( node_neg_complexity, "N1")
        
        location = NodeLocation(18,0,True,0)
        node = list.get_node( location )
        node_add_complexity, _ = get_quanta_impact( cfg, node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_quanta_impact( cfg, node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_quanta_impact( cfg, node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "S123")
        self.assertEqual( node_sub_complexity, "M123")
        self.assertEqual( node_neg_complexity, "")
        

if __name__ == '__main__':
    unittest.main()
