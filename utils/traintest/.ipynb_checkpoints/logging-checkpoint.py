import torch


class LoggedTensor(torch.Tensor):
    def __init__(self):
        self.buffer = []
        
    def concatenate(self):
        # This is done in a separate function rather than at each step because
        # otherwise the write performance would be horrible.
        # One has to be careful, when calling .mean or similar functions on the LoggedTensor.
        self.data = torch.cat(self.buffer, dim=0)
        
    def append(self, x):
        if type(x)!=torch.Tensor:
            x = torch.tensor(x)
        else:
            x = x.detach().cpu()
            
        if x.dim()==0:
            x = x.unsqueeze(0)
            
        self.buffer.append(x)
    
    
class Logger(dict):      
    def __init__(self, file=None, *args):
        dict.__init__(self, args)
        self.file = file
        
    def __getitem__(self, key):
        if key in self.keys():
            val = dict.__getitem__(self, key)
        else:
            val = LoggedTensor()
            self[key] = val
        return val
        
    def dump(self, file=None):
        self.concatenate()
        if file is None:
            file = self.file
        assert file is not None, "Logs can't be dumped if no file was specified"
        torch.save(self, file)
        
    def concatenate(self):
        for key in self:
            self[key].concatenate()
