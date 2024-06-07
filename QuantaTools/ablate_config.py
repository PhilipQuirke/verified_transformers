import transformer_lens.utils as utils

from .quanta_constants import NO_IMPACT_TAG 


# Intervention ablation configuration class
class AblateConfig():


    def __init__(self):
        self.reset_ablate()
        self.reset_intervention()
        self.reset_intervention_totals()
        self.operation = 0
        self.show_test_failures = False
        self.show_test_successes = False


    def reset_ablate_layer_store(self):
        # A list of "default" stored weightings collected from the model.
        self.layer_store = [[],[],[],[]]   # Supports 1 to 4 model layers


    def reset_ablate(self):
        self.threshold = 0.01

        # How many test questions are in the manually-curated varied_questions test set?
        self.num_varied_questions = 0
        # How many of the manually-curated varied_questions can the model answer?
        self.num_varied_successes = 0

        # attn.hook_z is the "attention head output" hook point name (at a specified layer)
        self.l_attn_hook_z_name = [utils.get_act_name('z', 0, 'a'),utils.get_act_name('z', 1, 'a'),utils.get_act_name('z', 2, 'a'),utils.get_act_name('z', 3, 'a')] # 'blocks.0.attn.hook_z' etc
        # hook_resid_pre is the "pre residual memory update" hook point name (at a specified layer)
        self.l_hook_resid_pre_name = ['blocks.0.hook_resid_pre','blocks.1.hook_resid_pre','blocks.2.hook_resid_pre','blocks.3.hook_resid_pre']
        # hook_resid_post is the "post residual memory update" hook point name (at a specified layer)
        self.l_hook_resid_post_name = ['blocks.0.hook_resid_post','blocks.1.hook_resid_post','blocks.2.hook_resid_post','blocks.3.hook_resid_post']
        # mlp.hook_post is the "MLP layer" hook point name (at a specified layer)
        self.l_mlp_hook_post_name = [utils.get_act_name('post', 0),utils.get_act_name('post', 1),utils.get_act_name('post', 2),utils.get_act_name('post', 3)] # 'blocks.0.mlp.hook_post' etc

        # A list of hooks that action the ablation interventions
        self.resid_put_hooks = [] # layer level
        self.attn_get_hooks = [] # node level
        self.attn_put_hooks = [] # node level
        
        # Sample model outputs used in ablation interventions
        self.mean_attn_z = []
        self.mean_resid_post = []
        self.mean_mlp_hook_post = []

        # A list of NodeLocations to ablate
        self.ablate_node_locations = []

        self.reset_ablate_layer_store()


    def reset_intervention(self, expected_answer = "", expected_impact = NO_IMPACT_TAG):

        # Expected output of an intervention ablation experiment
        self.expected_answer = expected_answer
        self.expected_impact = expected_impact if expected_impact != "" else NO_IMPACT_TAG

        # Actual outputs of an intervention ablation experiment
        self.intervened_answer = ""
        self.intervened_impact = NO_IMPACT_TAG

        # Auto-generated description of the ablation experiment 
        self.ablate_description = ""
        
        self.abort = False


    def reset_intervention_totals(self):
        self.num_filtered_nodes = 0
        self.num_tests_run = 0
        self.num_tags_added = 0


    @property
    def ablate_node_names(self) -> str:
        answer = ""

        for node in self.ablate_node_locations:
            answer += ( "" if answer == "" else ", " ) + node.name()          

        return answer


    def print_prediction_success_rate(self):
        bad_predictions = self.num_varied_questions - self.num_varied_successes

        if self.num_varied_questions > 0:
            print(f"Varied_questions prediction success rate = {self.num_varied_successes / self.num_varied_questions * 100:.2f}% ({self.num_varied_successes} good, {bad_predictions} bad)")

        if bad_predictions == 0:
            # This is evidence not proof because there may be very rare edge cases (say 1 in ten million) that do not exist in the test questions.
            # Even if you believe you know all the edge cases, and have enriched the training data to contain them, you may not have thought of all edge cases, so this is not proof.
            print("Model got all test questions correct.")
        else:
            print("WARNING: Model got", bad_predictions, "test questions wrong.")


# Global singleton class instance
acfg = AblateConfig()
