import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm

from .ablate_config import acfg
from .maths_data_generator import maths_data_generator
from .maths_test_questions import test_maths_questions_by_complexity, test_maths_questions_by_impact

# Calculate the per-token probability by comparing a batch of prediction "logits" to answer "tokens"
def logits_to_tokens_loss(cfg, logits, tokens):
    # Addition answer can have one extra digit than question. Answer also has a +/- sign
    n_answer_digits = cfg.num_answer_positions

    # The addition answer digit token probabilities
    ans_logits = logits[:, -(n_answer_digits+1):-1]

    # Convert raw score (logits) vector into a probability distribution.
    # Emphasizes the largest scores and suppress the smaller ones, to make them more distinguishable.
    ans_probs = F.log_softmax(ans_logits.to(torch.float64), dim=-1)

    max_prob_tokens = torch.argmax(ans_probs, dim=-1)

    # The addition answer digit tokens
    ans_tokens = tokens[:, -(n_answer_digits):]

    # Extract values from the ans_probs tensor, based on indices from the ans_tokens tensor
    ans_loss = torch.gather(ans_probs, -1, ans_tokens[:, :, None])[..., 0]

    return ans_loss, max_prob_tokens


# Calculate loss as negative of average per-token mean probability
def loss_fn(ans_loss):
    return -ans_loss.mean(0)


def one_million_questions_core(cfg):
  acfg.verbose = False

  cfg.analysis_seed = 345621 # Randomly chosen
  local_ds = maths_data_generator() # Re-initialise the data generator

  the_successes = 0
  the_fails = 0

  num_batches = 1000000//cfg.batch_size
  for epoch in tqdm(range(num_batches)):
      tokens = next(local_ds)

      the_fails = test_maths_questions_by_impact(cfg, acfg, tokens, 0, False)

      if the_fails> 0:
        break

      the_successes = the_successes + cfg.batch_size

      if epoch % 100 == 0:
          print("Batch", epoch, "of", num_batches, "#Successes=", the_successes)

  print("successes", the_successes, "num_fails", the_fails)
  if the_fails > 0:
    "WARNING: Model is not fully accurate. It failed the 1M Q test"


def one_million_questions(cfg):
  store_perc_sub = cfg.perc_sub
  store_perc_mult = cfg.perc_mult

  def print_config():
      print("%Mult=", cfg.perc_mult, "%Sub=", cfg.perc_sub, "%Add=", cfg.perc_add(), "File", cfg.file_config_prefix())

  print_config()
  print()

  if cfg.perc_add() > 0:
    print("Addition:")
    cfg.perc_sub = 0
    cfg.perc_mult = 0
    one_million_questions_core(cfg)

  if store_perc_sub > 0:
    print("Subtraction:")
    cfg.perc_sub = 100
    cfg.perc_mult = 0
    one_million_questions_core(cfg)
    print()

  cfg.perc_sub = store_perc_sub
  cfg.perc_mult = store_perc_mult
