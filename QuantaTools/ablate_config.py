import transformer_lens.utils as utils

from .quanta_type import NO_IMPACT_TAG 


class AblateConfig():


    def __init__(self):
        self.reset()
        self.reset_ablate_hooks()
        self.reset_intervention()
        self.reset_intervention_totals()
        self.show_test_failures = False


    def reset(self):
        self.threshold = 0.01
        self.verbose = False

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

        # Sample model outputs used in ablation interventions
        self.mean_attn_z = []
        self.mean_resid_post = []
        self.mean_mlp_hook_post = []


    def reset_ablate_hooks(self):

        # A list of "default" stored weightings collected from the model.
        # Same length as nodes
        self.layer_store = [[],[],[],[]]   # Supports 1 to 4 model layers

        # A list of hooks that action the ablation interventions
        self.attn_get_hooks = []
        self.attn_put_hooks = []
        self.resid_put_hooks = []

        # A list of NodeLocations to ablate
        self.ablate_node_locations = []
        # A specific input token position to ablate
        self.ablate_position = 0
        # A specific attention head to ablate
        self.ablate_attn_head = 0


    def reset_intervention(self, expected_answer = "", expected_impact = NO_IMPACT_TAG, operation = 0):
        self.operation = operation

        # Expected output of an intervention ablation experiment
        self.expected_answer = expected_answer
        self.expected_impact = expected_impact if expected_impact != "" else NO_IMPACT_TAG

        # Actual outputs of an intervention ablation experiment
        self.intervened_answer = ""
        self.intervened_impact = NO_IMPACT_TAG

        self.abort = False


    def reset_intervention_totals(self):
        self.num_tests_run = 0
        self.num_tags_added = 0


    def node_names(self):
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
            print("Model got all test questions correct. This is a pre-requisite for the model to be fully accurate, but this is NOT proof.")
        else:
            print("WARNING: Model is not fully accurate as it got", bad_predictions, "questions wrong.")
          


# Global singleton class instance
acfg = AblateConfig()
