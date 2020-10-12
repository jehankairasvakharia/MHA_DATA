import pandas as pd
from scipy.stats import beta
import numpy as np

def compute_assign_prob_TS(alpha_1, beta_1, alpha_2, beta_2, num_sims):
    """
    Return assign prob for action 1 and action 2
    """
    action_1_count = 0
    action_2_count = 0
    for i in range(num_sims):
        draw_1 = np.random.beta(alpha_1, beta_1)
        draw_2 = np.random.beta(alpha_2, beta_2)
        if draw_1 > draw_2:
            action_1_count +=1
        else:
            action_2_count +=1
            
    return action_1_count/num_sims, action_2_count/num_sims #arm 1 prob, arm2 prob

def compute_assign_prob_TS_2(alphas: list, betas: list, num_sims: int, versions: list):
    draw = np.random.beta(alphas, betas, (num_sims, len(alphas)))
    arms = list(np.argmax(draw, axis=1))
    return dict({'probability_{}'.format(versions[i]): arms.count(i) / num_sims for i in range(len(alphas))})