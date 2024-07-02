import random
import torch

from QuantaTools.model_token_to_char import tokens_to_string

from QuantaTools.quanta_constants import QType

from .maths_constants import MathsBehavior, MathsToken
from .maths_complexity import get_maths_question_complexity
from .maths_utilities import make_a_maths_question_and_answer


def maths_data_generator_start( cfg ):
    batch = torch.zeros((cfg.batch_size, cfg.n_ctx)).to(torch.int64)
    x = torch.randint(0, 10, (cfg.batch_size, cfg.n_digits))
    y = torch.randint(0, 10, (cfg.batch_size, cfg.n_digits))
    return (batch, x, y)


def maths_data_generator_mid( cfg, x, batch_op, y, batch ):
    batch[:, :cfg.n_digits] = x
    batch[:, cfg.n_digits] = batch_op
    batch[:, 1+cfg.n_digits:1+cfg.n_digits*2] = y
    batch[:, cfg.num_question_positions-1] = MathsToken.EQUALS

    # Convert each row into a 5-digit number
    x_values = x[:, 0]
    y_values = y[:, 0]
    for dn in range(1,cfg.n_digits):
        x_values = x_values * 10 + x[:, dn]
        y_values = y_values * 10 + y[:, dn]
        
    return (batch, x_values, y_values)        


def maths_data_generator_end( cfg, answers, batch ):

    # Insert the answers into the batch
    for i in range(cfg.batch_size):
        answer = answers[i]

        sign = MathsToken.PLUS
        if answer < 0:
            sign = MathsToken.MINUS
            answer = - answer

        batch[i, cfg.num_question_positions] = sign
        for j in range(cfg.n_digits+1):
            batch[i, cfg.n_ctx-j-1] = answer % 10
            answer = answer // 10
            if answer == 0:
                break

    return batch


# Generate an (optionally enriched) data batch for 
# "Addition" batch entries formated as XXXXX+YYYYY=+ZZZZZZ e.g. 550030+800020=+1350050
def maths_data_generator_addition( cfg, enrich_data=True ):

    (batch, x, y) = maths_data_generator_start( cfg )

    # Enrich the question data on 60% of batches to speed up training
    if enrich_data and (random.randint(1, 5) < 3):
        # Flatten x and y to 1D tensors
        x_flat = x.view(-1)
        y_flat = y.view(-1)

        # The UseSum9 task is compound and rare and so hard to learn.
        # Increase the MakeSum9 case frequency
        # UseSum9 also relies on MakeCarry1 (50%) from previous column.
        num_elements_to_modify = int(0.40 * x.numel()) # 40%
        indices_to_modify = torch.randperm(x_flat.numel())[:num_elements_to_modify]
        if random.randint(1, 2) == 1:
            x_flat[indices_to_modify] = 9 - y_flat[indices_to_modify]
        else:
            y_flat[indices_to_modify] = 9 - x_flat[indices_to_modify]

        # Reshape x and y back to its original shape
        x = x_flat.view(x.shape)
        y = y_flat.view(x.shape)

    (batch, x_values, y_values) = maths_data_generator_mid( cfg, x, MathsToken.PLUS, y, batch )
 
    # Elementwise operations to give the 1D tensor answers
    answers = x_values + y_values

    return maths_data_generator_end( cfg, answers, batch )


# Generate an (optionally enriched) data batch for  
# "Subtraction" batch entries formated as XXXXX-YYYYY=-ZZZZZZ e.g. 550030-800020=-0249990, 800020-550030=+0249990
def maths_data_generator_subtraction( cfg, enrich_data=True ):

    (batch, x, y) = maths_data_generator_start( cfg )

    # Enrich the question data on 60% of batches to speed up training
    if enrich_data and (random.randint(1, 5) < 3):
        # Flatten x and y to 1D tensors
        x_flat = x.view(-1)
        y_flat = y.view(-1)

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

    (batch, x_values, y_values) = maths_data_generator_mid( cfg, x, MathsToken.MINUS, y, batch )

    # Elementwise operations to give the 1D tensor answers
    answers = x_values - y_values

    return maths_data_generator_end( cfg, answers, batch )


# Generate an (optionally enriched) data batch for  
# "Multiplication" batch entries formated as 000XXX*000YYY=+ZZZZZZ e.g. 000345*000678=+233910
def maths_data_generator_multiplication( cfg, enrich_data=True ):

    (batch, x, y) = maths_data_generator_start( cfg )

    # Convert from NNNNNN*NNNNNN= to 000NNN*000NNN= so answer (product) is +0NNNNNN
    num_zeros = cfg.n_digits // 2
    for z in range(num_zeros):
        x[:, z] = 0
        y[:, z] = 0
        
    if enrich_data:
        # No data enrichment yet, but could be added in the future
        pass

    (batch, x_values, y_values) = maths_data_generator_mid( cfg, x, MathsToken.MULT, y, batch )

    # Elementwise operations to give the 1D tensor answers
    answers = x_values * y_values

    return maths_data_generator_end( cfg, answers, batch )


# Define "iterator" maths "questions" data generator function. Invoked using next().
# Generates an (optionally enriched) data batch containing ONE maths operation.
def maths_data_generator( cfg, enrich_data=True ):
    torch.manual_seed(cfg.analysis_seed)
    while True:

        batch_rand = random.randint(1, 100)
        if batch_rand <= cfg.perc_mult:
            batch = maths_data_generator_multiplication( cfg, enrich_data )
        elif batch_rand <= cfg.perc_mult + cfg.perc_sub:
            batch = maths_data_generator_subtraction( cfg, enrich_data )
        else:
            batch = maths_data_generator_addition( cfg, enrich_data )

        yield batch.cuda()
        

def maths_data_generator_mixed_core( cfg, enrich_data=True ):
    
    if cfg.perc_add == 100:
        return maths_data_generator_addition( cfg, enrich_data )
    elif cfg.perc_sub == 100:
        return maths_data_generator_subtraction( cfg, enrich_data )
    elif cfg.perc_mult == 100:
        return maths_data_generator_multiplication( cfg, enrich_data )
    else:
        # Assume a mixture of add and sub for now
        batch1 = maths_data_generator_addition( cfg, enrich_data )
        batch2 = maths_data_generator_subtraction( cfg, enrich_data )
       
        # Return a mixed batch with cfg.sub_perc % of subtraction questions, rest addition
        return torch.cat((batch1[:cfg.batch_size*cfg.perc_sub//100], batch2[:cfg.batch_size*(100-cfg.perc_sub)//100]), 0)
    
    

# Define "iterator" maths "questions" data generator function. Invoked using next().
# Generates a data batch for multiple maths operation.
def maths_data_generator_mixed( cfg, enrich_data=True ):
    torch.manual_seed(cfg.analysis_seed)
    while True:

        batch = maths_data_generator_mixed_core( cfg, enrich_data )

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