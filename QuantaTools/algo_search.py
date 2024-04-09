import itertools

from .useful_config import UsefulConfig
from .quanta_filter import FilterAlgo, filter_nodes
from .quanta_constants import QType, QCondition
from .algo_config import AlgoConfig



# Search the specified useful node(s), using the test_function, for the expected impact on the_impact_digit
def search_and_tag_digit_position(cfg, acfg, the_impact_digit, the_test_nodes, test_function, strong, the_tag, do_pair_search ):

    # Try single nodes first
    for node in the_test_nodes.nodes:
        if test_function(cfg, acfg, [node], the_impact_digit, strong):
            full_tag = the_tag + ("" if strong else "." + acfg.intervened_impact)
            acfg.num_tags_added += node.add_tag(QType.ALGO.value, full_tag)
            return True

    # Try pairs of nodes. Sometimes a task is split across two attention heads (i.e. a virtual attention head)
    if do_pair_search:
        node_pairs = list(itertools.combinations(the_test_nodes.nodes, 2))
        for pair in node_pairs:
            # Only if the 2 nodes are in the same layer can they can act in parallel and so "sum" to give a virtual attention head
            if pair[0].layer == pair[1].layer and pair[0].is_head == pair[1].is_head:
                if test_function(cfg, acfg, [pair[0], pair[1]], the_impact_digit, strong):
                    full_tag = the_tag + ("" if strong else "." + acfg.intervened_impact)
                    acfg.num_tags_added += pair[0].add_tag(QType.ALGO.value, full_tag)
                    acfg.num_tags_added += pair[1].add_tag(QType.ALGO.value, full_tag)
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
                the_filters = prerequisites_function(cfg, position, the_impact_digit)
                #print( "Filters:", the_filters.describe()) # Debugging
                
                # Filter useful nodes as per callers prerequisites
                test_nodes = filter_nodes( cfg.useful_nodes, the_filters)
                
                # Do not test nodes that already have the search tag assigned (perhaps from a previous search run)
                test_nodes = filter_nodes( test_nodes, FilterAlgo(the_tag, QCondition.NOT))

                acfg.num_filtered_nodes += len(test_nodes.nodes)
                
                if search_and_tag_digit_position(cfg, acfg, the_impact_digit, test_nodes, test_function, strong, the_tag, do_pair_search ):
                    return True

    return False


# For each answer digit, for each useful position, search the related useful node(s), using the test_function, for the expected impact on the_impact_digit. We may do 2 passes.
def search_and_tag(cfg, acfg, prerequisites_function, test_function, tag_function, do_pair_search, do_weak_search, from_position = -1, to_position = -1):
    acfg.reset_intervention_totals()

    for the_impact_digit in range(cfg.num_answer_positions):
        search_and_tag_digit(cfg, acfg, 
            prerequisites_function, the_impact_digit, test_function, tag_function,
            do_pair_search, do_weak_search, from_position, to_position )

    print(f"Filtering gave {acfg.num_filtered_nodes} candidate node(s). Ran {acfg.num_tests_run} intervention test(s). Added {acfg.num_tags_added} tag(s)")
