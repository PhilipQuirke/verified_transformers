import unittest

from QuantaTools.model_token_to_char import token_to_char, tokens_to_string

from QuantaTools.useful_node import NodeLocation, UsefulNode, UsefulNodeList

from QuantaTools.quanta_constants import QType, MATH_ADD_SHADES, MATH_SUB_SHADES
from QuantaTools.quanta_map_impact import sort_unique_digits, get_quanta_impact
from QuantaTools.quanta_map_attention import get_quanta_attention

from QuantaTools.ablate_config import acfg

from QuantaTools.maths_tools.maths_config import MathsConfig
from QuantaTools.maths_tools.maths_constants import MathsToken, MathsBehavior
from QuantaTools.maths_tools.maths_utilities import set_maths_vocabulary, int_to_answer_str, tokens_to_unsigned_int
from QuantaTools.maths_tools.maths_test_questions import make_maths_test_questions_and_answers, make_maths_questions_and_answers
from QuantaTools.maths_tools.maths_test_questions import make_maths_s0_questions_and_answers, make_maths_s1_questions_and_answers, make_maths_s2_questions_and_answers, make_maths_s3_questions_and_answers, make_maths_s4_questions_and_answers, make_maths_s5_questions_and_answers
from QuantaTools.maths_tools.maths_test_questions import make_maths_m0_questions_and_answers, make_maths_m1_questions_and_answers, make_maths_m2_questions_and_answers, make_maths_m3_questions_and_answers
from QuantaTools.maths_tools.maths_test_questions import make_maths_n1_questions_and_answers, make_maths_n2_questions_and_answers, make_maths_n3_questions_and_answers, make_maths_n4_questions_and_answers
from QuantaTools.maths_tools.maths_complexity import get_maths_min_complexity
from QuantaTools.maths_tools.maths_search_add import add_ss_test1
from QuantaTools.maths_tools.maths_search_sub import neg_nd_test1


class TestMaths(unittest.TestCase):

    def get_cfg(self):
        cfg = MathsConfig()
        cfg.n_digits = 6    
        cfg.perc_sub = 60
        cfg.use_cuda = False        
        set_maths_vocabulary(cfg)
        return cfg


    def test_int_to_answer_str(self):
        cfg = self.get_cfg()
        self.assertEqual( int_to_answer_str(cfg, 1234), "+0001234" )

        
    def test_tokens_to_unsigned_int(self):
        q = [0,1,2,3,4,5]
        offset = 0
        digits = 6
        self.assertEqual( tokens_to_unsigned_int(q, offset, digits), 12345 )


    def test_set_maths_vocabulary(self):
        cfg = self.get_cfg()

        self.assertEqual( token_to_char(cfg, 4), '4')
        self.assertEqual( token_to_char(cfg, MathsToken.MULT), '*')
        self.assertEqual( tokens_to_string(cfg, [MathsToken.EQUALS,4,0,7]), '=407')
  

    def test_sort_unique_digits(self):
        self.assertEqual( sort_unique_digits("A1231231278321", False), "12378")
        self.assertEqual( sort_unique_digits("A1231231278321", True), "87321")


    # During the construction of the test data, we check that the complexity of the question matches what the test data believes it is
    def test_make_maths_test_questions_and_answers(self):
        cfg = self.get_cfg()
        
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
        cfg = self.get_cfg()
        
        self.assertEqual( cfg.repeat_digit(4), 444444)


        
    # Intervention ablation test for addition "Use Sum 9" (SS) task
    def test_add_ss_test1(self):
        cfg = self.get_cfg()
        cfg.n_digits = 5 
        
        store_question, clean_question, intervened_answer = add_ss_test1(cfg, 4)

        self.assertEqual( store_question[0], 25222 )
        self.assertEqual( clean_question[0], 34633 )
        self.assertEqual( clean_question[0] + clean_question[1], 90188 )
        self.assertEqual( intervened_answer, 80188 )


    # Intervention ablation test for addition "Use Sum 9" (SS) task
    def test_neg_nd_test1(self):
        cfg = self.get_cfg()
        cfg.n_digits = 6 
        
        store_question, clean_question, intervened_answer = neg_nd_test1(cfg, 3)

        self.assertEqual( store_question[0], 33333 )
        self.assertEqual( clean_question[0], 99999 )
        self.assertEqual( clean_question[0] - clean_question[1], -344445 )
        self.assertEqual( intervened_answer, -347445 )


    def get_useful_node_list(self):
        cfg = self.get_cfg()
  
        the_list = UsefulNodeList()

        # Data from ins1_mix_d6_l3_h4_t40K_s372001 for P18L0
        
        the_locn = NodeLocation(18,0,True,0)
        the_list.add_node_tag( the_locn, QType.FAIL.value, '6' )
        the_list.add_node_tag( the_locn, QType.IMPACT.value, 'A2' )
        the_list.add_node_tag( the_locn, QType.MATH_ADD.value, 'S123' )
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'M123' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P4=50' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P11=50' )
        the_list.add_node_tag( the_locn, QType.MATH_ADD.value, 'A0.SP.Weak' ) 
        the_list.add_node_tag( the_locn, QType.MATH_ADD.value, 'A2.SP.Weak' )      
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'A0.MP.Weak' )      
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'A2.MP.Weak' )      
     
        the_locn = NodeLocation(18,0,True,1)
        the_list.add_node_tag( the_locn, QType.FAIL.value, '44' )
        the_list.add_node_tag( the_locn, QType.IMPACT.value, 'A2' )
        the_list.add_node_tag( the_locn, QType.MATH_ADD.value, 'S01234' )
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'M0123' )
        the_list.add_node_tag( the_locn, QType.MATH_NEG.value, 'N1234' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P3=52' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P10=48' )
        the_list.add_node_tag( the_locn, QType.MATH_ADD.value, 'A0.SP.Weak' ) 
        the_list.add_node_tag( the_locn, QType.MATH_ADD.value, 'A1.SP.Weak' )      
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'A0.MP.Weak' )      
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'A1.MP.Weak' )   

        the_locn = NodeLocation(18,0,True,2)
        the_list.add_node_tag( the_locn, QType.FAIL.value, '58' )
        the_list.add_node_tag( the_locn, QType.IMPACT.value, 'A2' )
        the_list.add_node_tag( the_locn, QType.MATH_ADD.value, 'S012345' )
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'M0123' )
        the_list.add_node_tag( the_locn, QType.MATH_NEG.value, 'N1234' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P10=51' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P3=47' )

        the_locn = NodeLocation(18,0,True,3)
        the_list.add_node_tag( the_locn, QType.FAIL.value, '12' )
        the_list.add_node_tag( the_locn, QType.IMPACT.value, 'A2' )
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'M0123' )
        the_list.add_node_tag( the_locn, QType.MATH_NEG.value, 'N1234' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P6=86' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P14=6' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P13=3' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P11=2' )
        the_list.add_node_tag( the_locn, QType.MATH_ADD.value, 'A2.SP' ) 
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'A0.MP.Weak' )      
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'A1.MP.Weak' )   
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'A2.MP.Weak' )      
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'A3.MP.Weak' )         
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'A4.MP.Weak' )      
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'A5.MP.Weak' )      
  
        the_locn = NodeLocation(18,0,False,0)
        the_list.add_node_tag( the_locn, QType.FAIL.value, '50' )
        the_list.add_node_tag( the_locn, QType.IMPACT.value, 'A2' )
        the_list.add_node_tag( the_locn, QType.MATH_ADD.value, 'S012345' )         
        the_list.add_node_tag( the_locn, QType.MATH_SUB.value, 'M0123' )
        the_list.add_node_tag( the_locn, QType.MATH_NEG.value, 'N1234' )

        return cfg, the_list
    

    def test_useful_node_list_complexity(self):
        
        cfg, the_list = self.get_useful_node_list()      
        
        the_locn = NodeLocation(18,0,True,0)
        the_node = the_list.get_node( the_locn )
        node_add_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "S1")
        self.assertEqual( node_sub_complexity, "M1")
        self.assertEqual( node_neg_complexity, "")

        the_locn = NodeLocation(18,0,True,1)
        the_node = the_list.get_node( the_locn )        
        node_add_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "S0")
        self.assertEqual( node_sub_complexity, "M0")
        self.assertEqual( node_neg_complexity, "N1")

        the_locn = NodeLocation(18,0,True,2)
        the_node = the_list.get_node( the_locn )        
        node_add_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "S0")
        self.assertEqual( node_sub_complexity, "M0")
        self.assertEqual( node_neg_complexity, "N1")

        the_locn = NodeLocation(18,0,True,3)
        the_node = the_list.get_node( the_locn )        
        node_add_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "")
        self.assertEqual( node_sub_complexity, "M0")
        self.assertEqual( node_neg_complexity, "N1")
        
        the_locn = NodeLocation(18,0,False,0)
        the_node = the_list.get_node( the_locn )        
        node_add_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_maths_min_complexity( cfg, the_node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "S0")
        self.assertEqual( node_sub_complexity, "M0")
        self.assertEqual( node_neg_complexity, "N1")
        
        the_locn = NodeLocation(18,0,True,0)
        the_node = the_list.get_node( the_locn )
        node_add_complexity, _ = get_quanta_impact( cfg, the_node, QType.MATH_ADD.value, MathsBehavior.ADD_COMPLEXITY_PREFIX.value, MATH_ADD_SHADES)
        node_sub_complexity, _ = get_quanta_impact( cfg, the_node, QType.MATH_SUB.value, MathsBehavior.SUB_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        node_neg_complexity, _ = get_quanta_impact( cfg, the_node, QType.MATH_NEG.value, MathsBehavior.NEG_COMPLEXITY_PREFIX.value, MATH_SUB_SHADES)
        self.assertEqual( node_add_complexity, "S123")
        self.assertEqual( node_sub_complexity, "M123")
        self.assertEqual( node_neg_complexity, "")
        

    def test_useful_node_list_save_load(self):
        
        cfg, the_list = self.get_useful_node_list()

        the_file_name = "test_useful_node_list.json"
        the_list.save_nodes(the_file_name)

        the_list2 = UsefulNodeList()  
        the_list2.load_nodes(the_file_name)   
        self.assertEqual( len(the_list.nodes), len(the_list2.nodes) )
        for i in range(len(the_list.nodes)):
            self.assertEqual( the_list.nodes[i].name(), the_list2.nodes[i].name() )
            self.assertEqual( len(the_list.nodes[i].tags), len(the_list2.nodes[i].tags) )


    # Test that two attention tags are sorted alphabetically if they are within 5% of each other
    def test_get_quanta_attention(self):
          
        cfg, the_list = self.get_useful_node_list()  
 
        the_locn = NodeLocation(18,0,True,2)
        the_node = the_list.get_node( the_locn )   
        
        cell_text, color_index = get_quanta_attention(cfg, the_node, QType.ATTN.value, "", 4)
        self.assertEqual( cell_text, "P10 P3") 
        
        the_list.reset_node_tags( QType.ATTN.value )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P10=51' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P3=47' )
        cell_text, color_index = get_quanta_attention(cfg, the_node, QType.ATTN.value, "", 4)
        self.assertEqual( cell_text, "P10 P3") 

        the_list.reset_node_tags( QType.ATTN.value )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P10=48' )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P3=47' )
        cell_text, color_index = get_quanta_attention(cfg, the_node, QType.ATTN.value, "", 4)
        self.assertEqual( cell_text, "P10 P3") 

        the_list.reset_node_tags( QType.ATTN.value )
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P3=47' ) # Reverse order
        the_list.add_node_tag( the_locn, QType.ATTN.value, 'P10=45' )
        cell_text, color_index = get_quanta_attention(cfg, the_node, QType.ATTN.value, "", 4)
        self.assertEqual( cell_text, "P10 P3") 

        
