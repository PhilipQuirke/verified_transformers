import itertools
from abc import ABC, abstractmethod

from .useful_config import UsefulConfig
from .quanta_constants import QType, QCondition
from .quanta_filter import FilterNode, FilterOr, FilterHead, FilterAlgo, FilterContains, FilterPosition, FilterImpact, FilterAlgo, filter_nodes
from .algo_config import AlgoConfig


class SubTaskBase(ABC):
    """
    Abstract base class for tasks, enforcing implementation of tag, prereqs, and test methods.
    """

    @staticmethod
    @abstractmethod
    def operation():
        """
        Method to return the operation assoicated with the task.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def tag(impact_digit):
        """
        Method to generate a tag for the task.
        """
        pass

    @staticmethod
    @abstractmethod
    def prereqs(cfg, position, impact_digit):
        """
        Method to calculate prerequisites for the task.
        """
        pass

    @staticmethod
    @abstractmethod
    def test(cfg, acfg, impact_digit, strong):
        """
        Method to test the task's implementation and interventions.
        """
        pass

    @staticmethod
    # A test function that always suceeds 
    def succeed_test(_, acfg, alter_digit, strong):
        print( "Test confirmed", acfg.ablate_node_names, "" if strong else "Weak")
        return True
    


# Search the specified useful node(s), using the test_function, for the expected impact on the_impact_digit
def search_and_tag_digit_position(cfg, acfg, the_impact_digit, test_nodes, sub_task_functions, strong, the_tag, do_pair_search ):

    success = False
    
    # Try single nodes first
    # For tasks like GT, SGN, OPR nodes at say 2 layers can satisfy the test.
    for node in test_nodes.nodes:
        acfg.ablate_node_locations = [node]
        if sub_task_functions.test(cfg, acfg, the_impact_digit, strong):
            full_tag = the_tag + ("" if strong else "." + acfg.intervened_impact)
            acfg.num_tags_added += node.add_tag(QType.ALGO.value, full_tag)
            success = True

    if not success:   
        # Try pairs of nodes. Sometimes a task is split across two attention heads (i.e. a virtual attention head)
        if do_pair_search:
            node_pairs = list(itertools.combinations(test_nodes.nodes, 2))
            for pair in node_pairs:
                # Only if the 2 nodes are in the same layer can they can act in parallel and so "sum" to give a virtual attention head
                if pair[0].layer == pair[1].layer and pair[0].is_head == pair[1].is_head:
                    acfg.ablate_node_locations = [pair[0], pair[1]]
                    if sub_task_functions.test(cfg, acfg, the_impact_digit, strong):
                        full_tag = the_tag + ("" if strong else "." + acfg.intervened_impact)
                        acfg.num_tags_added += pair[0].add_tag(QType.ALGO.value, full_tag)
                        acfg.num_tags_added += pair[1].add_tag(QType.ALGO.value, full_tag)
                        success = True

    return success


# For each useful position, search the related useful node(s), using the test_function, for the expected impact on the_impact_digit.
def search_and_tag_digit(cfg, acfg, sub_task_functions, the_impact_digit,
        do_pair_search : bool = False, # Search for "pairs" of interesting nodes (as well as "single" nodes) that satisfy the test \
        allow_impact_mismatch : bool = False, # Succeed in search even if expected impact is not correct
        delete_existing_tags : bool = True): # Delete existing tags before adding new ones

    the_tag = sub_task_functions.tag(the_impact_digit)

    if delete_existing_tags:
      # If a given search is run multiple times, the second run is impacted 
      # by the output of the first run via below code: FilterAlgo(the_tag, QCondition.NOT) 
      # Deleting the tags avoids this undesirable behavior.
      cfg.useful_nodes.reset_node_tags(QType.ALGO.value, the_tag)        

    from_position = cfg.min_useful_position
    to_position = cfg.max_useful_position

    # In some models, we don't predict the intervened_answer correctly in test_function.
    # So we may do a weak second pass 
    for strong in [True, False]:
        if strong or allow_impact_mismatch:

            # For tasks like GT, SGN, OPR nodes at say 2 positions can satisfy the test.
            success = False
            for position in range(from_position, to_position+1):
                the_filters = sub_task_functions.prereqs(cfg, position, the_impact_digit)
                
                # Filter useful nodes as per callers prerequisites
                test_nodes = filter_nodes( cfg.useful_nodes, the_filters)
                
                # Do not test nodes that already have the search tag assigned (perhaps from a previous search run)
                test_nodes = filter_nodes( test_nodes, FilterAlgo(the_tag, QCondition.NOT))

                num_test_nodes = len(test_nodes.nodes)
                acfg.num_filtered_nodes += num_test_nodes
                
                if num_test_nodes > 0 :
                    if search_and_tag_digit_position(cfg, acfg, the_impact_digit, test_nodes, sub_task_functions, strong, the_tag, do_pair_search ):
                        success = True
                    
            if success:
                return True

    return False


# For each answer digit, for each useful position, search the related useful node(s), using the test_function, for the expected impact on the_impact_digit. We may do 2 passes.
def search_and_tag(cfg, acfg, 
        sub_task_functions, 
        do_pair_search : bool = False, # Search for "pairs" of interesting nodes (as well as "single" nodes) that satisfy the test \
        allow_impact_mismatch : bool = False, # Succeed in search even if expected impact is not correct
        delete_existing_tags : bool = True): # Delete existing tags before adding new ones

    acfg.reset_intervention_totals()
    acfg.operation = sub_task_functions.operation()

    for the_impact_digit in range(cfg.num_answer_positions):
        search_and_tag_digit(cfg, acfg, 
            sub_task_functions, the_impact_digit, 
            do_pair_search=do_pair_search, 
            allow_impact_mismatch=allow_impact_mismatch, 
            delete_existing_tags=delete_existing_tags )

    print(f"Filtering gave {acfg.num_filtered_nodes} candidate node(s). Ran {acfg.num_tests_run} intervention test(s). Added {acfg.num_tags_added} tag(s)")
