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


    def set_model_names(self, model_names):
        super().set_model_names(model_names)
        self.reset_algo()


    def start_algorithm_test(self, acfg):
        self.reset_algo()

        acfg.print_prediction_success_rate()
        print()

        # Get the model nodes with a known algorithmic purpose
        return filter_nodes( self.useful_nodes, FilterAlgo("", QCondition.MUST) )


    # Does a useful node exist matching the filters? If so, return the position
    def test_algo_clause(self, node_list, the_filters, mandatory : bool = True):
        answer_position = -1
      
        matching_nodes = filter_nodes(node_list, the_filters)
        num_nodes = len(matching_nodes.nodes)

        if num_nodes > 0:
            print("Clause valid:", matching_nodes.node_names, " match", the_filters.describe())
            self.num_algo_valid_clauses += 1
            answer_position = matching_nodes.nodes[0].position
        elif mandatory:
            print("Clause invalid: No nodes match", the_filters.describe())
            self.num_algo_invalid_clauses += 1

        return answer_position


    def test_algo_logic(self, clause_name, clause_valid):

        if clause_valid:
            print("Clause valid:", clause_name)
            self.num_algo_valid_clauses += 1
        else:
            print("Clause invalid:", clause_name)
            self.num_algo_invalid_clauses += 1


    # Show the fraction of hypothesis clauses that were valid
    def print_algo_clause_results(self):
        print("Overall", self.num_algo_valid_clauses, "out of", self.num_algo_valid_clauses + self.num_algo_invalid_clauses, "algorithm clauses succeeded")
        

    # Show the fraction of useful nodes that have an assigned algorithmic purpose
    def print_algo_purpose_results(self, algo_nodes):
        num_heads = self.useful_nodes.num_heads
        num_neurons = self.useful_nodes.num_neurons

        num_heads_with_purpose = algo_nodes.num_heads
        num_neurons_with_purpose = algo_nodes.num_neurons

        print()
        if num_heads>0:
            print(f"{num_heads_with_purpose} of {num_heads} useful attention heads ({num_heads_with_purpose / num_heads * 100:.2f}%) have an algorithmic purpose assigned." )
        if num_neurons>0:
            print(f"{num_neurons_with_purpose} of {num_neurons} useful MLP neurons ({num_neurons_with_purpose / num_neurons * 100:.2f}%) have an algorithmic purpose assigned." )
