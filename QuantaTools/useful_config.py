import matplotlib.pyplot as plt

from .model_config import ModelConfig

from .useful_node import position_name, UsefulNodeList


# Extends ModelConfig with info on which layers and nodes (attention heads and MLP neuron) in the model are actually useful.
# Also refer https://github.com/PhilipQuirke/verified_transformers/blob/main/useful_tags_readme.md
class UsefulConfig(ModelConfig):
    
    # Create an empty useful configuration
    def __init__(self):
        super().__init__()
        self.reset_useful()
        

    # Reset the useful positions and nodes to empty
    def reset_useful(self):
        # Sparce ordered list of useful (question and answer) token positions actually used by the model e.g. 0,1,8,9,10,11
        self.useful_positions = []

        # List of the useful attention heads and MLP neurons that the model actually uses
        self.useful_nodes = UsefulNodeList()
 

    def set_model_names(self, model_names):
        super().set_model_names(model_names)
        self.reset_useful()


    @property  
    # Returns the minimum useful token position
    def min_useful_position(self) -> int:
        return min(self.useful_positions) if len(self.useful_positions) > 0 else -1


    @property
    # Returns the maximum useful token position
    def max_useful_position(self) -> int:
        return max(self.useful_positions) if len(self.useful_positions) > 0 else -1


    # Add a token position that we know is used in calculations
    def add_useful_position(self, position):
        if not (position in self.useful_positions):
            self.useful_positions += [position]


    # Add/update a useful node location, adding the specifying major and minor tags
    def add_useful_node_tag(self, the_location, major_tag : str, minor_tag : str ):
        assert the_location.position >= 0
        assert the_location.layer >= 0
        assert the_location.num >= 0
        assert the_location.position < self.n_ctx
        assert the_location.layer < self.n_layers
        if the_location.is_head:
            assert the_location.num < self.n_heads

        self.useful_nodes.add_node_tag( the_location, major_tag, minor_tag )


    # Show the positions, their meanings, and the number of questions that failed when that position is ablated in a 3 row table
    def calc_position_failures_map(self, num_failures_list, width_inches=16):
        columns = ["Posn"]
        for i in range(len(self.token_position_meanings)):
            columns += [position_name(i)]
    
        data = [
            ["Posn"] + self.token_position_meanings,
            ["# fails"] + num_failures_list
        ]
    
        _, ax = plt.subplots(figsize=(width_inches,1))
        ax.axis('tight')
        ax.axis('off')
    
        table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)  # Set the font size here
        table.scale(1, 1.5)  # The first parameter scales column widths, the second scales row heights
