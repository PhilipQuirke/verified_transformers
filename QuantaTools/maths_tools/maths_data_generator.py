import random
import torch

from QuantaTools.model_token_to_char import tokens_to_string

from QuantaTools.quanta_constants import QType

from .maths_constants import MathsBehavior, MathsToken
from .maths_complexity import get_maths_question_complexity
from .maths_utilities import make_a_maths_question_and_answer


# Generate an (optionally enriched) data batch for ONE maths operation selected from:
# "Addition" batch entries are formated XXXXX+YYYYY=+ZZZZZZ e.g. 550030+800020=+1350050
# "Subtraction" batch entries are formated XXXXX-YYYYY=-ZZZZZZ e.g. 550030-800020=-0249990, 800020-550030=+0249990
# "Multiplication" batch entries are formated 000XXX*000YYY=+ZZZZZZ e.g. 000345*000678=+233910
# Enrichment is used to speed up training by adding more complex cases to the training data
def maths_data_generator_single_core( cfg, batch_op, enrich_data=True ):

    batch = torch.zeros((cfg.batch_size, cfg.n_ctx)).to(torch.int64)
    x = torch.randint(0, 10, (cfg.batch_size, cfg.n_digits))
    y = torch.randint(0, 10, (cfg.batch_size, cfg.n_digits))

    if batch_op == MathsToken.MULT:
        # Convert from NNNNNN*NNNNNN= to 000NNN*000NNN= so answer (product) is +0NNNNNN
        num_zeros = cfg.n_digits // 2
        for z in range(num_zeros):
            x[:, z] = 0
            y[:, z] = 0

    # Enrich the question data on 60% of batches to speed up training
    if enrich_data and ( batch_op == MathsToken.PLUS or batch_op == MathsToken.MINUS ) and (random.randint(1, 5) < 3):
        # Flatten x and y to 1D tensors
        x_flat = x.view(-1)
        y_flat = y.view(-1)

        if batch_op == MathsToken.PLUS :
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
            if random.randint(1, 100) == 1:
                # For rare cases like 099111-099111=+0000000 some models predict -0000000. Generate more of these cases
                y_flat = x_flat.clone()

            else:
                # Empirically, the model seems to struggle with the sign calculation.
                # Minus signs are rarer than positive signs.
                # Generate more negative answers by increasing the y value
                y_flat[y_flat < 9] += 1

        # Reshape x and y back to its original shape
        x = x_flat.view(x.shape)
        y = y_flat.view(x.shape)


    first_answer_index = cfg.num_question_positions

    batch[:, :cfg.n_digits] = x
    batch[:, cfg.n_digits] = batch_op
    batch[:, 1+cfg.n_digits:1+cfg.n_digits*2] = y
    batch[:, first_answer_index-1] = MathsToken.EQUALS

    # Convert each row into a 5-digit number
    x_values = x[:, 0]
    y_values = y[:, 0]
    for dn in range(1,cfg.n_digits):
        x_values = x_values * 10 + x[:, dn]
        y_values = y_values * 10 + y[:, dn]

    # Elementwise operations to give the 1D tensor answers
    if batch_op == MathsToken.MULT:
        answers = x_values * y_values
    elif batch_op == MathsToken.MINUS:
        answers = x_values - y_values
    else:
        answers = x_values + y_values

    # Insert the answers into the batch
    for i in range(cfg.batch_size):
        answer = answers[i]

        sign = MathsToken.PLUS
        if answer < 0:
            sign = MathsToken.MINUS
            answer = - answer

        batch[i, first_answer_index] = sign
        for j in range(cfg.n_digits+1):
            batch[i, cfg.n_ctx-j-1] = answer % 10
            answer = answer // 10
            if answer == 0:
                break

    return batch


# Define "iterator" maths "questions" data generator function. Invoked using next().
# Generates an (optionally enriched) data batch for ONE maths operation.
def maths_data_generator( cfg, enrich_data=True ):
    torch.manual_seed(cfg.analysis_seed)
    while True:

        batch_rand = random.randint(1, 100)
        batch_op = MathsToken.MULT if batch_rand <= cfg.perc_mult else MathsToken.MINUS if batch_rand <= cfg.perc_mult + cfg.perc_sub else MathsToken.PLUS

        batch = maths_data_generator_single_core( cfg, batch_op, enrich_data )

        yield batch.cuda()
    

# Generate a data batch for multiple maths operation.
def maths_data_generator_mixed_core(cfg):

    batch = torch.zeros((cfg.batch_size, cfg.n_ctx)).to(torch.int64)
    x = torch.randint(0, 10, (cfg.batch_size, cfg.n_digits))
    y = torch.randint(0, 10, (cfg.batch_size, cfg.n_digits))

    # Generate a batch of random operation choices.
    # Currently ignores specific perc_sub and perc_mult values.
    if cfg.perc_mult > 0:
        operation_choices = [random.choice([MathsToken.PLUS, MathsToken.MINUS, MathsToken.MULT]) for _ in range(cfg.batch_size)]
    else:
        operation_choices = [random.choice([MathsToken.PLUS, MathsToken.MINUS]) for _ in range(cfg.batch_size)]

    first_answer_index = cfg.num_question_positions
    
    for i in range(cfg.batch_size):
        batch_op = operation_choices[i]
        batch[i, :cfg.n_digits] = x[i]
        batch[i, cfg.n_digits] = batch_op
        batch[i, 1 + cfg.n_digits:1 + cfg.n_digits * 2] = y[i]
        batch[i, first_answer_index - 1] = MathsToken.EQUALS

        x_values = x[i, 0]
        y_values = y[i, 0]
        for dn in range(1, cfg.n_digits):
            x_values = x_values * 10 + x[i, dn]
            y_values = y_values * 10 + y[i, dn]

        if batch_op == MathsToken.MULT:
            answer = x_values * y_values
        elif batch_op == MathsToken.MINUS:
            answer = x_values - y_values
        else:
            answer = x_values + y_values

        sign = MathsToken.PLUS
        if answer < 0:
            sign = MathsToken.MINUS
            answer = -answer

        batch[i, first_answer_index] = sign
        for j in range(cfg.n_digits + 1):
            batch[i, cfg.n_ctx - j - 1] = answer % 10
            answer = answer // 10
            if answer == 0:
                break

    return batch


# Define "iterator" maths "questions" data generator function. Invoked using next().
# Generates a data batch for multiple maths operation.
def maths_data_generator_mixed( cfg ):
    torch.manual_seed(cfg.analysis_seed)
    while True:

        batch = maths_data_generator_mixed_core( cfg )

        yield batch.cuda()
        

# Create a (matrix) batch of questions from a 2D matrix of ints
def make_maths_questions_and_answers(cfg, operator, major_tag, minor_tag, q_matrix):
    max_len = len(q_matrix)
    real_len = 0
    questions = torch.zeros((max_len, cfg.n_ctx)).to(torch.int64)
    limit = 10 ** cfg.n_digits

    for i in range(max_len):
        a = q_matrix[i][0]
        b = q_matrix[i][1]

        if a < limit and b < limit:
            make_a_maths_question_and_answer(cfg, questions, real_len, a, b, operator)

            good = True
            if (major_tag != QType.UNKNOWN) and (minor_tag != MathsBehavior.UNKNOWN ):
                # Check that the complexity of the question matches what the test data believes it is
                actual_major_tag, actual_minor_tag = get_maths_question_complexity(cfg, questions[real_len])
                question_str = tokens_to_string(cfg, questions[real_len])
                if not( actual_major_tag == major_tag and actual_minor_tag == minor_tag ):
                    print("make_maths_questions_and_answers complexity: Mismatch", question_str, major_tag.value, minor_tag.value, actual_major_tag.value, actual_minor_tag.value )
                    good = False

            if good:
                real_len += 1

    return questions[:real_len]