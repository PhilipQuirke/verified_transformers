from QuantaTools.useful_node import position_name, answer_name, UsefulNode, UsefulNodeList

from QuantaTools.quanta_constants import QType, QCondition, NO_IMPACT_TAG
from QuantaTools.quanta_map_impact import sort_unique_digits
from QuantaTools.quanta_filter import FilterNode, FilterAnd, FilterOr, FilterHead, \
     FilterContains, FilterPosition, FilterAttention, FilterImpact, FilterAlgo

from QuantaTools.ablate_config import AblateConfig

from .maths_constants import MathsToken, MathsBehavior, MathsTask 
from .maths_search_mix import succeed_test, math_common_prereqs, \
    run_strong_intervention, run_weak_intervention
from .maths_utilities import digit_name

    
# Tag for addition "Use Sum 9" (SS) task e.g. 34633+55555=+090188 where D4 and D'4 sum to 9 (4+5), and D3 + D'3 > 10
def add_ss_tag(impact_digit):
    return answer_name(impact_digit-1)  + "." + MathsTask.SS_TAG.value


# Node rerequisites for addition "Use Sum 9" (SS) task
def add_ss_prereqs(cfg, position, impact_digit):
    # Pays attention to Dn-2 and D'n-2. Impacts An
    return math_common_prereqs(cfg, position, impact_digit-2, impact_digit)


def add_ss_test1(cfg, alter_digit):
    # 25222 + 44444 = 69666. Has no Dn-2.SC but has Dn-1.SS so not a UseSum9 case
    store_question = [cfg.repeat_digit(2), cfg.repeat_digit(4)]
    store_question[0] += (5-2) * 10 ** (alter_digit - 1)

    # 34633 + 55555 = 90188. Has Dn-2.SC and Dn-1.SS so is a UseSum9 case
    clean_question = [cfg.repeat_digit(3), cfg.repeat_digit(5)]
    clean_question[0] += (4-3) * 10 ** (alter_digit - 1)
    clean_question[0] += (6-3) * 10 ** (alter_digit - 2)

    # When we intervene we expect answer 80188
    intervened_answer = clean_question[0] + clean_question[1] - 10 ** (alter_digit)

    return store_question, clean_question, intervened_answer


# Intervention ablation test for addition "Use Sum 9" (SS) task
def add_ss_test(cfg, acfg, alter_digit, strong):
    if alter_digit < 2 or alter_digit > cfg.n_digits:
        acfg.reset_intervention()
        return False

    store_question, clean_question, intervened_answer = add_ss_test1(cfg, alter_digit)

    success, _, _ = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

    if success:
        print( "Test confirmed", acfg.ablate_node_names(), "perform", add_ss_tag(alter_digit), "impacting", intervention_impact+" accuracy.", "" if strong else "Weak")

    return success


# Tag for addition "Make Carry 1" (SC) task e.g. 222222+666966=+0889188 where D2 + D'2 > 10
def add_sc_tag(impact_digit):
    return answer_name(impact_digit-1)  + "." + MathsTask.SC_TAG.value


# Node rerequisites for addition "Make Carry 1" (SC) task
def add_sc_prereqs(cfg, position, impact_digit):
    # Pays attention to Dn-1 and D'n-1. Impacts An
    return math_common_prereqs(cfg, position, impact_digit-1, impact_digit)


# Intervention ablation test for addition "Make Carry 1" (SC) task
def add_sc_test(cfg, acfg, impact_digit, strong):
    alter_digit = impact_digit - 1

    if alter_digit < 0 or alter_digit >= cfg.n_digits:
        acfg.reset_intervention()
        return False

    intervention_impact = answer_name(impact_digit)

    # 222222 + 666966 = 889188. Has Dn.SC
    store_question = [cfg.repeat_digit(2), cfg.repeat_digit(6)]
    store_question[1] += (9 - 6) * (10 ** alter_digit)

    # 333333 + 555555 = 888888. No Dn.SC
    clean_question = [cfg.repeat_digit(3), cfg.repeat_digit(5)]

    # When we intervene we expect answer 889888
    intervened_answer = clean_question[0] + clean_question[1] + 10 ** (alter_digit+1)

    success, _, _ = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

    if success:
        print( "Test confirmed", acfg.ablate_node_names(), "perform", add_sc_tag(alter_digit), "impacting", intervention_impact, " accuracy.", "" if strong else "Weak")

    return success


# Tag for addition "Simple Add" (SA) task e.g. 555555+111111=+0666666 where D3 + D'3 < 10
def add_sa_tag(impact_digit):
    return answer_name(impact_digit) + "." + MathsTask.SA_TAG.value


# Node rerequisites for addition "Simple Add" (SA) task
def add_sa_prereqs(cfg, position, impact_digit):
    # Pays attention to Dn and D'n. Impacts An
    return math_common_prereqs(cfg, position, impact_digit, impact_digit)


def add_sa_test1(cfg, alter_digit):
    # 222222 + 111111 = +333333. No Dn.SC
    store_question = [cfg.repeat_digit(2), cfg.repeat_digit(1)]

    # 555555 + 444444 = +999999. No Dn.SC
    clean_question = [cfg.repeat_digit(5), cfg.repeat_digit(4)]

    # When we intervene we expect answer +999399
    intervened_answer = clean_question[0] + clean_question[1] + (3-9) * 10 ** alter_digit

    return store_question, clean_question, intervened_answer


def add_sa_test2(cfg, alter_digit):
    # 222222 + 666666 = +888888. No Dn.SC
    store_question = [cfg.repeat_digit(2), cfg.repeat_digit(6)]

    # 555555 + 111111 = +666666. No Dn.SC
    clean_question = [cfg.repeat_digit(5), cfg.repeat_digit(1)]

    # When we intervene we expect answer +666866
    intervened_answer = clean_question[0] + clean_question[1] + (8-6) * 10 ** alter_digit

    return store_question, clean_question, intervened_answer


# Intervention ablation test for addition "Simple Add" (SA) task
def add_sa_test(cfg, acfg, alter_digit, strong):
    # Note: MD and SA give the same result when D'=0 or D=D'=5. We avoid ablation tests like this.

    intervention_impact = answer_name(alter_digit)

    store_question, clean_question, intervened_answer = add_sa_test1(cfg, alter_digit)
    success1, _, impact_success1 = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

    store_question, clean_question, intervened_answer = add_sa_test2(cfg, alter_digit)
    success2, _, impact_success2 = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

    success = (success1 and success2) if strong else (impact_success1 and impact_success2)

    if success:
        print( "Test confirmed:", acfg.ablate_node_names(), "perform", add_sa_tag(alter_digit), "= (D"+str(alter_digit)+" + D'"+str(alter_digit)+") % 10 impacting "+intervention_impact+" accuracy.", "" if strong else "Weak", acfg.intervened_answer)

    return success


# Tag for addition An.ST 
def add_st_tag(focus_digit):
    return digit_name(focus_digit) + "." + MathsTask.ST_TAG.value


# Prerequisites for addition An.ST 
def add_st_prereqs(cfg, position, focus_digit):
    return FilterAnd(
        FilterHead(),
        FilterPosition(position_name(cfg.n_digits), QCondition.MIN), # Occurs from the operator token
        FilterPosition(position_name(cfg.num_question_positions), QCondition.MAX), # Occurs by the = token
        FilterAttention(cfg.dn_to_position_name(focus_digit)), # Attends to Dn
        FilterAttention(cfg.ddn_to_position_name(focus_digit)), # Attends to D'n
        FilterContains(QType.MATH_ADD, MathsBehavior.ADD_PCA_TAG.value), # Node PCA is interpretable (bigram or trigram output) with respect to addition T8,T9,T10
        FilterContains(QType.MATH_ADD, MathsBehavior.ADD_COMPLEXITY_PREFIX.value), # Impacts addition questions
        FilterPosition(position_name(position))) # Is at token position Px


# Intervention ablation test for addition An.ST with impact "A65432" to "A65" in early tokens.
def add_st_test(cfg, acfg, focus_digit, strong):
    # 222222 + 777977 = 1000188. Has Dn.SC
    store_question = [cfg.repeat_digit(2), cfg.repeat_digit(7)]
    store_question[1] += (9 - 7) * (10 ** focus_digit)

    # 333333 + 666666 = 999999. No Dn.SC
    clean_question = [cfg.repeat_digit(3), cfg.repeat_digit(6)]

    success = run_weak_intervention(cfg, acfg, store_question, clean_question)

    if success:
        description = acfg.ablate_node_names() + " perform " + add_st_tag(focus_digit) + " = TriCase(D"+str(focus_digit)+" + D'"+str(focus_digit)+")"
        print("Test confirmed", description, "Impact:", acfg.intervened_impact, "" if strong else "Weak")

    return success
