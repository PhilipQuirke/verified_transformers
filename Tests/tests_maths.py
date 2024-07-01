import torch
import unittest

from QuantaTools.model_token_to_char import token_to_char, tokens_to_string

from QuantaTools.useful_node import NodeLocation, UsefulNode, UsefulNodeList

from QuantaTools.quanta_constants import QType, MATH_ADD_SHADES, MATH_SUB_SHADES
from QuantaTools.quanta_map_impact import sort_unique_digits, get_quanta_impact
from QuantaTools.quanta_map_attention import get_quanta_attention

from QuantaTools.maths_tools.maths_config import MathsConfig
from QuantaTools.maths_tools.maths_constants import MathsToken, MathsBehavior
from QuantaTools.maths_tools.maths_utilities import set_maths_vocabulary, int_to_answer_str, tokens_to_unsigned_int
from QuantaTools.maths_tools.maths_data_generator import maths_data_generator_single_core, make_maths_questions_and_answers, maths_data_generator_mixed_core
from QuantaTools.maths_tools.maths_test_questions import make_maths_s0_questions_and_answers, make_maths_s1_questions_and_answers, make_maths_s2_questions_and_answers, make_maths_s3_questions_and_answers, make_maths_s4_questions_and_answers, make_maths_s5_questions_and_answers
from QuantaTools.maths_tools.maths_test_questions import make_maths_m0_questions_and_answers, make_maths_m1_questions_and_answers, make_maths_m2_questions_and_answers, make_maths_m3_questions_and_answers
from QuantaTools.maths_tools.maths_test_questions import make_maths_n1_questions_and_answers, make_maths_n2_questions_and_answers, make_maths_n3_questions_and_answers, make_maths_n4_questions_and_answers
from QuantaTools.maths_tools.maths_complexity import get_maths_min_complexity
from QuantaTools.maths_tools.maths_search_mix import (
    run_intervention_core, run_strong_intervention, run_weak_intervention,
    opr_functions, sgn_functions)
from QuantaTools.maths_tools.maths_search_add import (
    add_ss_functions, add_sc_functions, add_sa_functions, add_st_functions)
from QuantaTools.maths_tools.maths_search_sub import (
    sub_mt_functions, sub_gt_functions, sub_md_functions, sub_mb_functions, neg_nd_functions, neg_nb_functions)


class TestMaths(unittest.TestCase):

    def get_cfg(self):
        cfg = MathsConfig()
        cfg.n_digits = 6    
        cfg.perc_sub = 60
        cfg.use_cuda = False     
        cfg.sanity_check()
        set_maths_vocabulary(cfg)
        return cfg


    def test_to_dict(self):
        cfg = self.get_cfg()    
        data = cfg.to_dict()    
        self.assertEqual( data['n_layers'], 3)
        self.assertEqual( data['n_heads'], 4)
        self.assertEqual( data['perc_sub'], 60)
        self.assertEqual( data['n_digits'], 6)
        

    def test_init_from_json(self):
        cfg = self.get_cfg()    
        data = cfg.to_dict()   
        cfg.init_from_json(data)
        cfg.sanity_check()
        
        self.assertEqual( cfg.n_layers, 3)
        self.assertEqual( cfg.n_heads, 4)
        self.assertEqual( cfg.perc_sub, 60)
        self.assertEqual( cfg.n_digits, 6)        


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
    
        
    def test_maths_data_generator_core(self):
        
        cfg = self.get_cfg()
        cfg.perc_mult = 33
        cfg.perc_sub = 33
        
        torch.manual_seed(cfg.analysis_seed)
        questions = maths_data_generator_single_core(cfg, MathsToken.PLUS, True )
        questions = maths_data_generator_single_core(cfg, MathsToken.MINUS, True)
        questions = maths_data_generator_single_core(cfg, MathsToken.MULT, True )
        questions = maths_data_generator_single_core(cfg, MathsToken.PLUS, False )
        questions = maths_data_generator_single_core(cfg, MathsToken.MINUS, False )
        questions = maths_data_generator_single_core(cfg, MathsToken.MULT, False )

  
    def test_maths_data_generator_mixed(self):
        
        cfg = self.get_cfg()

        cfg.perc_mult = 33
        cfg.perc_sub = 33
        torch.manual_seed(cfg.analysis_seed)
        questions = maths_data_generator_mixed_core(cfg)
        
        cfg.perc_mult = 0
        cfg.perc_sub = 50
        torch.manual_seed(cfg.analysis_seed)
        questions = maths_data_generator_mixed_core(cfg)

        # print( tokens_to_string(cfg, questions[0]) )
        # print( tokens_to_string(cfg, questions[1]) )
        # print( tokens_to_string(cfg, questions[2]) )
        # print( tokens_to_string(cfg, questions[3]) )
        # print( tokens_to_string(cfg, questions[4]) )
  

    def test_repeat_digit(self):
        cfg = self.get_cfg()
        
        self.assertEqual( cfg.repeat_digit(4), 444444)


    def test_parse_model_name(self):
        cfg = self.get_cfg()
        cfg.model_name = "ins1_mix_d10_l3_h5_t50K_s572091"
        cfg.parse_model_name()
        self.assertEqual( cfg.insert_mode, 1)
        self.assertEqual( cfg.n_digits, 10)
        self.assertEqual( cfg.n_layers, 3)
        self.assertEqual( cfg.n_heads, 5)
        self.assertEqual( cfg.n_training_steps, 50000)
        self.assertEqual( cfg.training_seed, 572091)

        
    def test_parse_insert_model_name(self):
        cfg = self.get_cfg()
        cfg.parse_insert_model_name("add_d7_l6_h5_t40K_s572077")
        self.assertEqual( cfg.insert_n_digits, 7)
        self.assertEqual( cfg.insert_n_layers, 6)
        self.assertEqual( cfg.insert_n_heads, 5)
        self.assertEqual( cfg.insert_n_training_steps, 40000)
        self.assertEqual( cfg.insert_training_seed, 572077)
        self.assertEqual( cfg.grokfast, False)


    def test_parse_model_name_grokfast(self):
        cfg = self.get_cfg()
        
        cfg.model_name = "ins1_mix_d10_l3_h5_t50K_s572091"        
        cfg.parse_model_name()
        self.assertEqual( cfg.grokfast, False)
        self.assertEqual( cfg.file_config_prefix, cfg.model_name )      
        
        cfg.model_name = "ins1_mix_d10_l3_h5_t50K_gf_s572091"        
        cfg.parse_model_name()
        self.assertEqual( cfg.grokfast, True)       
        self.assertEqual( cfg.file_config_prefix, cfg.model_name)      


    # Intervention ablation test for addition "Use Sum 9" (SS) task
    def test_add_ss_test1(self):
        cfg = self.get_cfg()
        cfg.n_digits = 5 
        
        store_question, clean_question, intervened_answer = add_ss_functions.test1(cfg, 4)

        self.assertEqual( store_question[0], 25222 )
        self.assertEqual( clean_question[0], 34633 )
        self.assertEqual( clean_question[0] + clean_question[1], 90188 )
        self.assertEqual( intervened_answer, 80188 )


    # Intervention ablation test for addition "Use Sum 9" (SS) task
    def test_neg_nd_test1(self):
        cfg = self.get_cfg()
        cfg.n_digits = 6 
        
        store_question, clean_question, intervened_answer = neg_nd_functions.test1(cfg, 3)

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
 

    def test_useful_node_list_counts(self):
        
        cfg, the_list = self.get_useful_node_list()      
        
        assert(the_list.num_heads > 0)
        assert(the_list.num_neurons > 0)
        assert(the_list.node_names != "")
        

    def test_useful_node_list_get_node_by_tag(self):
        
        cfg, the_list = self.get_useful_node_list()      
        
        node = the_list.get_node_by_tag(QType.MATH_ADD.value, 'S012345' )
        assert( node.name() == "P18L0H2" )
        node = the_list.get_node_by_tag(QType.FAIL.value, '12' ) 
        assert( node.name() == "P18L0H3" )


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
    def test_get_quanta_attention_sorting(self):
          
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

        
    # Test the math subtask search function
    def test_get_math_subtask_search(self):
            
        cfg, the_list = self.get_useful_node_list() 
      
        tag = opr_functions.tag(2)
        filters = opr_functions.prereqs(cfg, 14, 2)
         
        tag = sgn_functions.tag(2)
        filters = sgn_functions.prereqs(cfg, 14, 2)  
        
        tag = add_ss_functions.tag(2)
        filters = add_ss_functions.prereqs(cfg, 14, 2)     
        
        tag = add_sc_functions.tag(2)
        filters = add_sc_functions.prereqs(cfg, 14, 2)    
                
        tag = add_sa_functions.tag(2)
        filters = add_sa_functions.prereqs(cfg, 14, 2)    
                
        tag = add_st_functions.tag(2)
        filters = add_st_functions.prereqs(cfg, 14, 2)    
                
        tag = sub_mt_functions.tag(2)
        filters = sub_mt_functions.prereqs(cfg, 14, 2)    
                
        tag = sub_gt_functions.tag(2)
        filters = sub_gt_functions.prereqs(cfg, 14, 2)   

        tag = sub_md_functions.tag(2)
        filters = sub_md_functions.prereqs(cfg, 14, 2)    
                
        tag = sub_mb_functions.tag(2)
        filters = sub_mb_functions.prereqs(cfg, 14, 2)    
                
        tag = neg_nd_functions.tag(2)
        filters = neg_nd_functions.prereqs(cfg, 14, 2)    
                
        tag = neg_nb_functions.tag(2)
        filters = neg_nb_functions.prereqs(cfg, 14, 2)    

