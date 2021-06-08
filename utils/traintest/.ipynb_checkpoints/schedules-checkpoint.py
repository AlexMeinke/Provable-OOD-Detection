import numpy as np
        
    
def create_piecewise_constant_schedule(lrs, drops):
    schedule = lrs[0]*np.ones(drops[-1])
    for idx, lr in enumerate(lrs[1:]):
        schedule[drops[idx]:] = lr
    return schedule


def create_piecewise_linear_schedule(lrs, drops):
    drops = [0] + list(drops)
    assert len(drops)==len(lrs), "Schedule specification invalid"
    schedule = np.concatenate([np.linspace(lrs[i], lrs[i+1], drops[i+1]-drops[i]) for i in range(len(lrs)-1)], 0)
    return schedule


def create_piecewise_exp_schedule(lrs, drops):
    drops = [0] + list(drops)
    assert len(drops)==len(lrs), "Schedule specification invalid"
    schedule = np.concatenate([np.geomspace(lrs[i], lrs[i+1], drops[i+1]-drops[i]) for i in range(len(lrs)-1)], 0)
    return schedule


schedule_dict = {
                    'constant': create_piecewise_constant_schedule,
                    'linear': create_piecewise_linear_schedule,
                    'exp': create_piecewise_exp_schedule
                }