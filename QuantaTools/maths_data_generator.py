import numpy as np
import random
import torch

from .maths_vocab import MathsTokens


# Generate an enriched data batch for one operator type
# "Addition" batch entries are formated XXXXX+YYYYY=+ZZZZZZ e.g. 550030+800020=+1350050
# "Subtraction" batch entries are formated XXXXX-YYYYY=-ZZZZZZ e.g. 550030-800020=-0249990, 800020-550030=+0249990
# "Multiplication" batch entries are formated 000XXX*000YYY=+ZZZZZZ e.g. 000345*000678=+233910
def data_generator_core( cfg, batch_op ):

  batch = torch.zeros((cfg.batch_size, cfg.n_ctx())).to(torch.int64)
  x = torch.randint(0, 10, (cfg.batch_size, cfg.n_digits))
  y = torch.randint(0, 10, (cfg.batch_size, cfg.n_digits))

  if batch_op == MathsTokens.MULT:
    # Convert from NNNNNN*NNNNNN= to 000NNN*000NNN= so answer (product) is NNNNNN
    num_zeros = cfg.n_digits // 2
    for z in range(num_zeros):
      x[:, z] = 0
      y[:, z] = 0

  # Enrich the question data on 60% of batches to speed up training
  if ( batch_op == MathsTokens.PLUS or batch_op == MathsTokens.MINUS ) and (random.randint(1, 5) < 3):
    # Flatten x and y to 1D tensors
    x_flat = x.view(-1)
    y_flat = y.view(-1)

    if batch_op == MathsTokens.PLUS :
      # The UseSum9 task is compound and rare and so hard to learn.
      # Increase the MakeSum9 case frequency
      # UseSum9 also relies on MakeCarry1 (50%) from previous column.
      num_elements_to_modify = int(0.40 * x.numel()) # 40%
      indices_to_modify = torch.randperm(x_flat.numel())[:num_elements_to_modify]
      if random.randint(1, 2) == 1:
        x_flat[indices_to_modify] = 9 - y_flat[indices_to_modify]
      else:
        y_flat[indices_to_modify] = 9 - x_flat[indices_to_modify]
    else:
      # Empirically, the model seems to struggle with the sign calculation.
      # Minus signs are rarer than positive signs.
      # Generate more negative answers by increasing the y value
      y_flat[y_flat < 9] += 1

    # Reshape x and y back to its original shape
    x = x_flat.view(x.shape)
    y = y_flat.view(x.shape)


  first_answer_index = cfg.question_tokens()

  batch[:, :cfg.n_digits] = x
  batch[:, cfg.n_digits] = batch_op
  batch[:, 1+cfg.n_digits:1+cfg.n_digits*2] = y
  batch[:, first_answer_index-1] = MathsTokens.EQUALS

  # Convert each row into a 5-digit number
  x_values = x[:, 0]
  y_values = y[:, 0]
  for dn in range(1,cfg.n_digits):
    x_values = x_values * 10 + x[:, dn]
    y_values = y_values * 10 + y[:, dn]

  # Elementwise operations to give the 1D tensor answers
  if batch_op == MathsTokens.MULT:
    answers = x_values * y_values
  else:
    if batch_op == MathsTokens.MINUS:
      answers = x_values - y_values
    else:
      answers = x_values + y_values

  # Insert the answers into the batch
  for i in range(cfg.batch_size):
    answer = answers[i]

    sign = MathsTokens.PLUS
    if answer < 0:
      sign = MathsTokens.MINUS
      answer = - answer

    batch[i, first_answer_index] = sign
    for j in range(cfg.n_digits+1):
      batch[i, cfg.n_ctx()-j-1] = answer % 10
      answer = answer // 10
      if answer == 0:
          break

  return batch


# Define "iterator" data generator function. Invoked using next().
def data_generator( cfg ):
  torch.manual_seed(cfg.analysis_seed)
  while True:

    batch_rand = random.randint(1, 100)
    batch_op = MathsTokens.MULT if batch_rand <= cfg.perc_mult else MathsTokens.MINUS if batch_rand <= cfg.perc_mult + cfg.perc_sub else MathsTokens.PLUS

    batch = data_generator_core( cfg, batch_op )

    yield batch.cuda()