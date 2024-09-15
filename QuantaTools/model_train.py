import numpy as np
import torch
import torch.nn.functional as F



# Calculate the per-token probability by comparing a batch of prediction "logits" to answer "tokens"
def logits_to_tokens_loss(cfg, logits, tokens):
    # Addition answer can have one extra digit than question. Answer also has a +/- sign
    n_answer_digits = cfg.num_answer_positions

    # The addition answer digit token probabilities
    # The "-1" below is needed because each answer digit calculations occurs one token before the that answer digit's token is revealed.
    ans_logits = logits[:, (-n_answer_digits-1):-1]

    # Convert raw score (logits) vector into a probability distribution.
    # Emphasizes the largest scores and suppress the smaller ones, to make them more distinguishable.
    ans_probs = F.log_softmax(ans_logits.to(torch.float64), dim=-1)

    max_prob_tokens = torch.argmax(ans_probs, dim=-1)

    # The addition answer digit tokens
    ans_tokens = tokens[:, (-n_answer_digits):]

    # Extract values from the ans_probs tensor, based on indices from the ans_tokens tensor
    ans_loss = torch.gather(ans_probs, -1, ans_tokens[:, :, None])[..., 0]

    return ans_loss, max_prob_tokens


# Calculate loss as negative of average per-token mean probability
def loss_fn(ans_loss):
    return -ans_loss.mean(0)


def get_training_optimizer_and_scheduler(cfg):
    optimizer = torch.optim.AdamW(cfg.main_model.parameters(),
                            lr = cfg.lr,
                            weight_decay = cfg.weight_decay,
                            betas = (0.9, 0.98))

    max_iter = cfg.n_training_steps
    warmup_iter = max_iter // 5
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=int(warmup_iter))
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(np.ceil((max_iter-warmup_iter))))
    scheduler  = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[int(warmup_iter)])
    
    return optimizer, scheduler