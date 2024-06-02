from QuantaTools.useful_node import position_name, answer_name, UsefulNode, UsefulNodeList
from QuantaTools.quanta_constants import QType, QCondition, NO_IMPACT_TAG
from QuantaTools.quanta_map_impact import sort_unique_digits
from QuantaTools.quanta_filter import ( FilterNode, FilterAnd, FilterOr, FilterHead,
     FilterContains, FilterPosition, FilterAttention, FilterImpact )
from QuantaTools.ablate_config import AblateConfig
from QuantaTools.ablate_hooks import a_predict_questions, a_run_attention_intervention

from .maths_constants import MathsToken, MathsBehavior, MathsTask 
from .maths_search_mix import run_intervention_core, run_strong_intervention, run_weak_intervention, SubTaskBaseMath
from .maths_utilities import digit_name
from .maths_config import MathsConfig


# Subtraction "Essential Borrow Info" (MT) sub-task. 
# Found in early tokens. Node output is tricase: MT+, MT0, MT-    
class sub_mt_functions(SubTaskBaseMath):

    @staticmethod
    def operation():
        return MathsToken.MINUS
    
    @staticmethod
    def tag(impact_digit):
        return answer_name(impact_digit) + "." + MathsTask.MT_TAG.value

    @staticmethod
    def prereqs(cfg, position, focus_digit):
        # Example meaning: 
        #   And(IsHead, 
        #       Position>=n_digits, Position<=num_question_positions, Position=14,
        #       AttendsTo:D3, AttendsTo:D'3, 
        #       MAY have bi- or trigram PCA for subtraction questions,
        #       Impacts subtraction questions)        
        return FilterAnd(
            FilterHead(),
            FilterPosition(position_name(cfg.n_digits), QCondition.MIN), # Occurs in early tokens
            FilterPosition(position_name(cfg.num_question_positions), QCondition.MAX), # Occurs in early tokens   
            FilterPosition(position_name(position)),
            FilterAttention(cfg.dn_to_position_name(focus_digit)), # Attends to Dn
            FilterAttention(cfg.ddn_to_position_name(focus_digit)), # Attends to D'n
            FilterContains(QType.MATH_SUB, MathsBehavior.SUB_PCA_TAG.value, QCondition.MAY), # Weak: Node PCA is interpretable (bigram or trigram output) with respect to subtraction ST8,ST9,ST10
            FilterContains(QType.MATH_SUB, MathsBehavior.SUB_COMPLEXITY_PREFIX.value)) # Impacts positive-answer subtraction questions (cover M1 to M4)
            # FilterContains(QType.MATH_SUB, MathsBehavior.NEG_COMPLEXITY_PREFIX.value)) # Impacts ngative-answer subtraction questions (cover N1 to N4)

    @staticmethod
    def test(cfg, acfg, focus_digit, strong):

        if focus_digit >= cfg.n_digits:
            acfg.reset_intervention()
            return False

        # 555555 - 000000 = +0555555. Is a positive-answer-subtraction
        store_question = [cfg.repeat_digit(5), cfg.repeat_digit(0)]

        # 222222 - 222422 = -0000200. Is a negative-answer-subtraction question because of focus_digit
        clean_question = [cfg.repeat_digit(2), cfg.repeat_digit(2)]
        clean_question[1] += 2 * (10 ** focus_digit)

        success = run_weak_intervention(cfg, acfg, store_question, clean_question)

        if success:
            print("Test confirmed", acfg.ablate_node_names(), "perform", sub_mt_functions.tag(focus_digit), "Impact:", acfg.intervened_impact, "" if strong else "Weak")

        return success


# Subtraction "Greater than or equal to" (GT) sub-task. 
# Found in early tokens. Node output is bicase: GT+, GT-    
class sub_gt_functions(SubTaskBaseMath):

    @staticmethod
    def operation():
        return MathsToken.MINUS
    
    @staticmethod
    # Tag for Greater Than "Dn > D'n" (Dn.GT) task used in SUB and NEG
    def tag(impact_digit):
      return answer_name(impact_digit) + "." + MathsTask.GT_TAG.value

    @staticmethod
    # Prerequisites for Greater Than "Dn > D'n" (Dn.GT) task used in SUB and NEG
    def prereqs(cfg, position, attend_digit):
        return FilterAnd(
            FilterHead(), # Is an attention head
            FilterPosition(position_name(position)), # Is at token position Px
            FilterAttention(cfg.dn_to_position_name(attend_digit)), # Attends to Dn
            FilterAttention(cfg.ddn_to_position_name(attend_digit))) # Attends to D'n

    @staticmethod
    # Intervention ablation test for subtraction "Dn > D'n" (Dn.GT) task
    def test(cfg, acfg, impact_digit, strong):
        impact_locn = (10 ** impact_digit)
        tail_nines = MathsConfig.repeat_digit_n(9,min(cfg.n_digits,impact_digit+2))

        # 00600-00201=+000399. SUB
        sub_question = [6 * impact_locn, 2 * impact_locn+1]
        # 00100-00201=-000101. NEG
        neg_question = [1 * impact_locn, 2 * impact_locn+1]

        # We expect the sign (Amax) to change from - to +
        intervention_impact = answer_name(cfg.n_digits+1)
        # Addition of 2 6-digit numbers can give a 7-digit answer. Subtraction of 2 6-digit numbers gives a signed 6-digit number. Exclude Amax-1
        digit = cfg.n_digits - 1
        while digit > impact_digit+1:
            intervention_impact += str(digit)
            digit -= 1  


        # TEST CHANGE FROM NEGATIVE TO POSITIVE ANSWER
        # 00600-00201=+000399. SUB
        store_question = [sub_question[0], sub_question[1]]
        # 00100-00201=-000101. NEG
        clean_question = [neg_question[0], neg_question[1]]
        clean_answer = clean_question[0] - clean_question[1]
        assert(clean_answer<0) # Negative clean answer
        expected_answer = + cfg.repeat_digit(9) - tail_nines - clean_answer 
        assert(expected_answer>=0) # Positive ablated answer
        run_intervention_core(cfg, acfg, store_question, clean_question, intervention_impact, expected_answer, strong=False)
        success = (acfg.intervened_answer[0] == "+")

        if acfg.show_test_failures and not success:
            print("Failed: GT_1, impact_digit:", impact_digit, acfg.ablate_description)
        if acfg.show_test_successes and success:
            print("Success: GT_1, " + acfg.ablate_description)


        if success:
          # TEST CHANGE FROM POSITIVE TO NEGATIVE ANSWER
            # 00100-00201=-000101. NEG
            store_question = [neg_question[0], neg_question[1]]
            # 00600-00201=+000399. SUB
            clean_question = [sub_question[0], sub_question[1]]
            clean_answer = clean_question[0] - clean_question[1] # positive
            assert(clean_answer>=0) # Positive clean answer
            # When we intervene we expect answer to swap from negative to positive. Get -0999101 
            expected_answer = - cfg.repeat_digit(9) + tail_nines - clean_answer 
            assert(expected_answer<0) # Negative ablated answer
            run_intervention_core(cfg, acfg, store_question, clean_question, intervention_impact, expected_answer, strong=False)
            success = (acfg.intervened_answer[0] == "-")

            #if acfg.show_test_failures and not success:
            if not success:
              print("Failed: GT_2, impact_digit:", impact_digit, acfg.ablate_description)
            if acfg.show_test_successes and success:
                print("Success: GT_2 ," + acfg.ablate_description)

 
        if success:
            print( "Test confirmed", acfg.ablate_node_names(), "perform", sub_gt_functions.tag(impact_digit))

        return success


# Positive-answer subtraction "Difference" (MD) tasks e.g. 666666-222222=+0444444 where D3 >= D'3
# Node output is one of 10 values 0 to 9
class sub_md_functions(SubTaskBaseMath):

    @staticmethod
    def operation():
        return MathsToken.MINUS
    
    @staticmethod
    def tag(impact_digit):
        return answer_name(impact_digit) + "." + MathsTask.MD_TAG.value

    @staticmethod
    def prereqs(cfg, position, impact_digit):
        # Pays attention to Dn and D'n. Impacts An
        return SubTaskBaseMath.math_latetoken_subtask_prereqs(cfg, position, impact_digit, impact_digit)

    @staticmethod
    def test1(cfg, alter_digit):
        # 333333 - 111111 = +222222. No Dn.MB
        store_question = [cfg.repeat_digit(3), cfg.repeat_digit(1)]

        # 999999 - 444444 = +555555. No DN.MB
        clean_question = [cfg.repeat_digit(9), cfg.repeat_digit(4)]

        # When we intervene we expect answer +555255
        intervened_answer = clean_question[0] - clean_question[1] + (2-5) * 10 ** alter_digit

        return store_question, clean_question, intervened_answer

    @staticmethod
    def test2(cfg, alter_digit):
        # 666666 - 222222 = +444444. No DN.MB
        store_question = [cfg.repeat_digit(6), cfg.repeat_digit(2)]

        # 999999 - 333333 = +666666. No DN.MB
        clean_question = [cfg.repeat_digit(9), cfg.repeat_digit(3)]

        # When we intervene we expect answer +666466
        intervened_answer = clean_question[0] - clean_question[1] + (4-6) * 10 ** alter_digit

        return store_question, clean_question, intervened_answer

    @staticmethod
    def test(cfg, acfg, alter_digit, strong):
        # Note: MD and SA give the same result when D'=0 or D=D'=5. We avoid ablation tests like this.
    
        intervention_impact = answer_name(alter_digit)

        store_question, clean_question, intervened_answer = sub_md_functions.test1(cfg, alter_digit)
        success1, _, impact_success1 = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        store_question, clean_question, intervened_answer = sub_md_functions.test2(cfg, alter_digit)
        success2, _, impact_success2 = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        success = (success1 and success2) if strong else (impact_success1 and impact_success2)

        if success:
            print( "Test confirmed", acfg.ablate_node_names(), "perform", sub_md_functions.tag(alter_digit), " = (D"+str(alter_digit)+" + D'"+str(alter_digit)+") % 10 impacting "+intervention_impact+" accuracy.", "" if strong else "Weak")

        return success


# Positive-answer subtraction "Borrow One" (MB) task e.g. 222222-111311=+0110911 where D2 > D'2
# Node output is binary (aka boolean)        
class sub_mb_functions(SubTaskBaseMath):

    @staticmethod
    def operation():
        return MathsToken.MINUS
    
    @staticmethod
    def tag(impact_digit):
        return answer_name(impact_digit-1)  + "." + MathsTask.MB_TAG.value    

    @staticmethod
    def prereqs(cfg, position, impact_digit):
        # Pays attention to Dn-1 and D'n-1. Impacts An    
        return SubTaskBaseMath.math_latetoken_subtask_prereqs(cfg, position, impact_digit-1, impact_digit)

    @staticmethod
    def test(cfg, acfg, impact_digit, strong):
        alter_digit = impact_digit - 1

        if alter_digit < 0 or alter_digit >= cfg.n_digits:
            acfg.reset_intervention()
            return False

        intervention_impact = answer_name(impact_digit)

        # 222222 - 111311 = +0110911. Has Dn.MB
        store_question = [cfg.repeat_digit(2), cfg.repeat_digit(1)]
        store_question[1] += (3 - 1) * (10 ** alter_digit)

        # 777777 - 444444 = +0333333. No Dn.MB
        clean_question = [cfg.repeat_digit(7), cfg.repeat_digit(4)]

        # When we intervene we expect answer +0332333
        intervened_answer = clean_question[0] - clean_question[1] - 10 ** (alter_digit+1)

        success, _, _ = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        if success:
            print( "Test confirmed", acfg.ablate_node_names(), "perform", sub_mb_functions.tag(alter_digit), "impacting", intervention_impact, "accuracy.", "" if strong else "Weak")
        
        return success


# Negative-answer subtraction "Difference" (ND) tasks e.g. 666666-928222=-0261556 where D3 >= D'3
# Node output is one of 10 values 0 to 9
class neg_nd_functions(SubTaskBaseMath):

    @staticmethod
    def operation():
        return MathsToken.MINUS
    
    @staticmethod
    def tag(impact_digit):
        return answer_name(impact_digit) + "." + MathsTask.ND_TAG.value

    @staticmethod
    def prereqs(cfg, position, impact_digit):
        # Impacts An and pays attention to Dn and D'n
        return SubTaskBaseMath.math_latetoken_subtask_prereqs(cfg, position, impact_digit, impact_digit)

    @staticmethod
    def test1(cfg, alter_digit):
        # 033333 - 111111 = -077778. No Dn.NB
        store_question = [cfg.repeat_digit(3), cfg.repeat_digit(1)]
        store_question[0] = store_question[0] // 10 # Convert 333333 to 033333

        # 099999 - 444444 = -344445. No Dn.NB
        clean_question = [cfg.repeat_digit(9), cfg.repeat_digit(4)]
        clean_question[0] = clean_question[0] // 10 # Convert 999999 to 099999

        # When we intervene we expect answer -347445
        intervened_answer = clean_question[0] - clean_question[1] - (7-4) * 10 ** alter_digit

        return store_question, clean_question, intervened_answer

    @staticmethod
    def test2(cfg, alter_digit):
        # 066666 - 222222 = -155556. No Dn.NB
        store_question = [cfg.repeat_digit(6), cfg.repeat_digit(2)]
        store_question[0] = store_question[0] // 10 # Remove top digit

        # 099999 - 333333 = -233334. No Dn.NB
        clean_question = [cfg.repeat_digit(9), cfg.repeat_digit(3)]
        clean_question[0] = clean_question[0] // 10 # Remove top digit

        # When we intervene we expect answer -231334
        intervened_answer = clean_question[0] - clean_question[1] - (5-3) * 10 ** alter_digit

        return store_question, clean_question, intervened_answer

    @staticmethod
    def test(cfg, acfg, alter_digit, strong):
        intervention_impact = answer_name(alter_digit)

        store_question, clean_question, intervened_answer = neg_nd_functions.test1(cfg, alter_digit)
        success1, _, impact_success1 = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        store_question, clean_question, intervened_answer = neg_nd_functions.test2(cfg, alter_digit)
        success2, _, impact_success2 = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        success = (success1 and success2) if strong else (impact_success1 and impact_success2)

        if success:
            print( "Test confirmed", acfg.ablate_node_names(), "perform", neg_nd_functions.tag(alter_digit), " = (D"+str(alter_digit)+" + D'"+str(alter_digit)+") % 10 impacting", intervention_impact, "accuracy.", "" if strong else "Weak")

        return success


# Negative-answer subtraction "Borrow One" (NB) task 
# Node output is binary (aka boolean)   
class neg_nb_functions(SubTaskBaseMath):

    @staticmethod
    def operation():
        return MathsToken.MINUS
    
    @staticmethod
    def tag(impact_digit):
        return answer_name(impact_digit-1) + "." + MathsTask.NB_TAG.value

    @staticmethod
    def prereqs(cfg, position, impact_digit):
        # Pays attention to Dn-1 and D'n-1. Impacts An
        return SubTaskBaseMath.math_latetoken_subtask_prereqs(cfg, position, impact_digit-1, impact_digit)

    @staticmethod
    def test(cfg, acfg, impact_digit, strong):
        alter_digit = impact_digit - 1

        if alter_digit < 0 or alter_digit >= cfg.n_digits:
            acfg.reset_intervention()
            return False

        intervention_impact = answer_name(impact_digit)

        # 022222 - 111311 = -0089089. Has Dn.MB
        store_question = [cfg.repeat_digit(2), cfg.repeat_digit(1)]
        store_question[0] = store_question[0] // 10 # Convert 222222 to 022222
        store_question[1] += (3 - 1) * (10 ** alter_digit)

        # 077777 - 444444 = -0366667. No Dn.MB
        clean_question = [cfg.repeat_digit(7), cfg.repeat_digit(4)]
        clean_question[0] = clean_question[0] // 10 # Convert 777777 to 077777

        # When we intervene we expect answer -0366677
        intervened_answer = clean_question[0] - clean_question[1] - 10 ** (alter_digit+1)

        success, _, _ = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        if success:
            print( "Test confirmed", acfg.ablate_node_names(), "perform", neg_nb_functions.tag(alter_digit), "impacting", intervention_impact, "accuracy.", "" if strong else "Weak")

        return success