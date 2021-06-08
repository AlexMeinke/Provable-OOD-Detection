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

    
class PiecewiseConstantSchedule:
    def __init__(self, values, drops):
        self.values = values
        self.drops = drops
    
    def __getitem__(self, epoch):
        if epoch < self.drops[0]:
            return self.values[0]
        else:
            for idx, drop in enumerate(self.drops):
                if epoch < drop:
                    return self.values[idx]
            else:
                raise IndexError(f'Epoch {epoch} higher then this schedules max epoch {drops[-1]-1}')
    
class ConstantSchedule:
    def __init__(self, one_value_list, drops=None):
        assert len(one_value_list) == 1
        self.value = one_value_list[0]
        
    def __getitem__(self, epoch):
        return self.value


schedule_dict = {   
                    'constant': create_piecewise_constant_schedule,
                    'PiecewiseConstant': PiecewiseConstantSchedule,
                    'Constant': ConstantSchedule,
                    'CompletelyConstant': ConstantSchedule,
                    'linear': create_piecewise_linear_schedule,
                    'exp': create_piecewise_exp_schedule
                }