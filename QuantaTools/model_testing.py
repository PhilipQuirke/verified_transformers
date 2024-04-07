from tqdm.notebook import tqdm
from .ablate_config import acfg
from .maths_data_generator import maths_data_generator
from .maths_test_questions import test_maths_questions_by_impact

def test_correctness_on_num_questions_core(cfg, num_questions=1000000):
  acfg.verbose = False

  cfg.analysis_seed = 345621 # Randomly chosen
  local_ds = maths_data_generator(cfg=cfg) # Re-initialise the data generator

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


def test_correctness_on_num_questions(cfg, num_questions=1000000):
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
    test_correctness_on_num_questions_core(cfg, num_questions=num_questions)

  if store_perc_sub > 0:
    print("Subtraction:")
    cfg.perc_sub = 100
    cfg.perc_mult = 0
    test_correctness_on_num_questions_core(cfg)
    print()

  cfg.perc_sub = store_perc_sub
  cfg.perc_mult = store_perc_mult
