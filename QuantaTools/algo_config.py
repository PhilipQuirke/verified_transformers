import itertools

from .useful_config import UsefulConfig
from .quanta_filter import FilterAlgo, filter_nodes
from .quanta_constants import QType, QCondition
 


# Extends UsefulConfig with algorithm functionality
class AlgoConfig(UsefulConfig):


    def __init__(self):
        super().__init__()
        self.reset_algo()
        

    def reset_algo(self):
        self.num_algo_valid_clauses = 0
        self.num_algo_invalid_clauses = 0
      
        
    def start_algorithm_test(self, acfg):
        self.reset_algo()

        acfg.print_prediction_success_rate()
        print()

        # Get the model nodes with a known algorithmic purpose
        return filter_nodes( self.useful_nodes, FilterAlgo("", QCondition.MUST) )
  

    # Does a useful node exist matching the filters? If so, return the position
    def test_algo_clause(self, node_list, the_filters):
        answer_position = -1
      
        matching_nodes = filter_nodes(node_list, the_filters)
        num_nodes = len(matching_nodes.nodes)

        if num_nodes > 0:
            print( "Clause valid:", matching_nodes.get_node_names(), " match", the_filters.describe())
            self.num_algo_valid_clauses += 1
            answer_position = matching_nodes.nodes[0].position
        else:
            print( "Clause invalid: No nodes match", the_filters.describe())
            self.num_algo_invalid_clauses += 1

        return answer_position
  

    def test_algo_logic(self, clause_name, clause_valid):

        if clause_valid:
            print( "Clause valid:", clause_name)
            self.num_algo_valid_clauses += 1
        else:
            print( "Clause invalid:", clause_name)
            self.num_algo_invalid_clauses += 1
        

    # Show the fraction of hypothesis clauses that were valid
    def print_algo_clause_results(self):
        print( "Overall", self.num_algo_valid_clauses, "out of", self.num_algo_valid_clauses + self.num_algo_invalid_clauses, "algorithm clauses succeeded")
        

    # Show the fraction of useful nodes that have an assigned algorithmic purpose
    def print_algo_purpose_results(self, algo_nodes):
        num_heads = self.useful_nodes.num_heads()
        num_neurons = self.useful_nodes.num_neurons()

        num_heads_with_purpose = algo_nodes.num_heads()
        num_neurons_with_purpose = algo_nodes.num_neurons()

        print()
        print( f"{num_heads_with_purpose} of {num_heads} useful attention heads ({num_heads_with_purpose / num_heads * 100:.2f}%) have an algorithmic purpose assigned." )
        print( f"{num_neurons_with_purpose} of {num_neurons} useful MLP neurons ({num_neurons_with_purpose / num_neurons * 100:.2f}%) have an algorithmic purpose assigned." )
        


# Search the specified useful node(s), using the test_function, for the expected impact on the_impact_digit
def search_and_tag_digit_position(acfg, the_impact_digit, the_test_nodes, test_function, strong, the_tag, do_pair_search ):

    # Try single nodes first
    for node in the_test_nodes.nodes:
        if test_function( [node], the_impact_digit, strong):
            full_tag = the_tag + ("" if strong else "." + acfg.intervened_impact)
            node.add_tag(QType.ALGO.value, full_tag)
            acfg.num_tags_added += 1
            return True

    # Try pairs of nodes. Sometimes a task is split across two attention heads (i.e. a virtual attention head)
    if do_pair_search:
        node_pairs = list(itertools.combinations(the_test_nodes.nodes, 2))
        for pair in node_pairs:
            # Only if the 2 nodes are in the same layer can they can act in parallel and so "sum" to give a virtual attention head
            if pair[0].layer == pair[1].layer and pair[0].is_head == pair[1].is_head:
                if test_function( [pair[0], pair[1]], the_impact_digit, strong):
                    full_tag = the_tag + ("" if strong else "." + acfg.intervened_impact)
                    pair[0].add_tag(QType.ALGO.value, full_tag)
                    pair[1].add_tag(QType.ALGO.value, full_tag)
                    acfg.num_tags_added += 2
                    return True

    return False


# For each useful position, search the related useful node(s), using the test_function, for the expected impact on the_impact_digit.
def search_and_tag_digit(cfg, acfg, prerequisites_function, the_impact_digit, test_function, tag_function, do_pair_search, do_weak_search, from_position, to_position ):

    the_tag = tag_function(the_impact_digit)

    if from_position == -1:
        from_position = cfg.min_useful_position()
    if to_position == -1:
        to_position = cfg.max_useful_position()

    # In some models, we don't predict the intervened_answer correctly in test_function.
    # So we may do a weak second pass and may add say "A5.BS.A632" tag to a node.
    for strong in [True, False]:
        if strong or do_weak_search:

            for position in range(from_position, to_position+1):
                test_nodes = filter_nodes( cfg.useful_nodes, prerequisites_function(position, the_impact_digit))

                acfg.num_filtered_nodes += len(test_nodes.nodes)
                
                if search_and_tag_digit_position(acfg,  the_impact_digit, test_nodes, test_function, strong, the_tag, do_pair_search ):
                    return True

    return False


# For each answer digit, for each useful position, search the related useful node(s), using the test_function, for the expected impact on the_impact_digit. We may do 2 passes.
def search_and_tag(cfg, acfg, prerequisites_function, test_function, tag_function, do_pair_search, do_weak_search, from_position = -1, to_position = -1):
    acfg.reset_intervention_totals()

    for the_impact_digit in range(cfg.num_answer_positions):
        search_and_tag_digit(cfg, acfg, 
            prerequisites_function, the_impact_digit, test_function, tag_function,
            do_pair_search, do_weak_search, from_position, to_position )
    print( "Filters are:", prerequisites_function.describe())
    print( f"Filtering gave {acfg.num_filtered_nodes} candidate node(s). Ran {acfg.num_tests_run} intervention test(s). Added {acfg.num_tags_added} tag(s)")