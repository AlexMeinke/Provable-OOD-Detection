import torch
import utils.traintest.evaluation as ev


class IDEvaluator():
    def __init__(self, model, device, in_loader, from_logits=True):
        self.model = model
        self.device = device
        self.in_loader = in_loader
        self.from_logits = from_logits
        
    def run(self, logged_scalars={}, logged_vectors={}, logged_img={}):
        print('Running ID eval...')
        self.model.eval()
        acc, conf_in = ev.get_accuracy(self.model, self.device, self.in_loader, self.from_logits)
        confusion_matrix, confidence_matrix = ev.get_confusion_matrix(self.model, self.device, self.in_loader, self.from_logits)
        
        logged_scalars['test/ID_MMC'] = conf_in.mean()
        
        logged_scalars['test/ID_Acc'] = acc
        
        logged_img['confusion_matrix/confusion'] = confusion_matrix.unsqueeze(0)
        logged_img['confusion_matrix/confidence'] = confidence_matrix.unsqueeze(0)
        
        class_wise_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
        for acc, class_name in zip(class_wise_acc, self.in_loader.dataset.classes):
            logged_scalars['class_wise_acc/' + class_name] = acc
            

class OODEvaluator():
    def __init__(self, model, device, in_loader, out_loader_dict, eps=0.01, use_ebm=False):
        self.model = model
        self.device = device
        self.in_loader = in_loader
        self.out_loader_dict = out_loader_dict
        self.eps = eps
        self.use_ebm = use_ebm
        
    def run(self, logged_scalars={}, logged_vectors={}, logged_img={}):
        print('Running OOD eval...')
        self.model.eval()
        
        outputs_in = ev.get_output(self.model, self.device, self.in_loader)
        log_conf_in = ev.get_log_conf(outputs_in)
        
        if self.use_ebm:
            neg_energy_in = ev.get_neg_energy(outputs_in)
        
        for out_dset in self.out_loader_dict:
            print('\t' + out_dset)
            loader = self.out_loader_dict[out_dset]
            
            outputs_out = ev.get_output(self.model, self.device, loader)
            log_conf_out = ev.get_log_conf(outputs_out)
            
            logged_scalars['test/' + out_dset + '_MMC'] = log_conf_out.exp().mean()
            
            auc = ev.auroc(log_conf_in.numpy(), log_conf_out.numpy())
            logged_scalars['test/' + out_dset + '_AUC'] = auc
            
            cauc = ev.auroc_conservative(log_conf_in.numpy(), log_conf_out.numpy())
            logged_scalars['test/' + out_dset + '_cAUC'] = cauc
            
            if hasattr(self.model, 'ibp_forward'):
                try:
                    ub = ev.get_ub_log_conf(self.model, self.device, loader, eps=self.eps, num_classes=10)
                except:
                    ub = ev.get_output_ub(self.model, self.device, loader, eps=self.eps).squeeze().numpy()
                gauc = ev.auroc_conservative(log_conf_in.numpy(), ub)
                logged_scalars['test/' + out_dset + '_GAUC'] = gauc
            
            if self.use_ebm:
                neg_energy_out = ev.get_neg_energy(outputs_out)
                logged_scalars['test_ebm/' + out_dset + '_mean_energy'] = (-neg_energy_out).mean()
            
                auc = ev.auroc(neg_energy_in.numpy(), neg_energy_out.numpy())
                logged_scalars['test_ebm/' + out_dset + '_AUC'] = auc

                cauc = ev.auroc_conservative(neg_energy_in.numpy(), neg_energy_out.numpy())
                logged_scalars['test_ebm/' + out_dset + '_cAUC'] = cauc

                
                ub = ev.get_ub_neg_energy(self.model, self.device, loader, eps=self.eps, num_classes=10)
                gauc = ev.auroc_conservative(neg_energy_in.numpy(), ub)
                logged_scalars['test_ebm/' + out_dset + '_GAUC'] = gauc

                
class BinaryEvaluator():
    def __init__(self, model, device, in_loader, out_loader_dict, eps=0.01):
        self.model = model
        self.device = device
        self.in_loader = in_loader
        self.out_loader_dict = out_loader_dict
        self.eps = eps
        
    def run(self, logged_scalars={}, logged_vectors={}, logged_img={}):
        print('Running OOD eval...')
        self.model.eval()
        
        outputs_in = ev.get_output(self.model, self.device, self.in_loader)
        log_conf_in = outputs_in[:,0]
        logged_scalars['test/ID_MMC'] = log_conf_in.mean()
        
        for out_dset in self.out_loader_dict:
            print('\t' + out_dset)
            loader = self.out_loader_dict[out_dset]
            
            outputs_out = ev.get_output(self.model, self.device, loader)
            if outputs_out.squeeze().dim()>1:
                log_conf_out = outputs_out.max(1)[0]
            else:
                log_conf_out = outputs_out[:,0]
            
            logged_scalars['test/' + out_dset + '_MMC'] = log_conf_out.exp().mean()
            
            auc = ev.auroc(log_conf_in.numpy(), log_conf_out.numpy())
            logged_scalars['test/' + out_dset + '_AUC'] = auc
            
            cauc = ev.auroc_conservative(log_conf_in.numpy(), log_conf_out.numpy())
            logged_scalars['test/' + out_dset + '_cAUC'] = cauc
            
            ub = ev.get_output_ub(self.model, self.device, loader, eps=self.eps)
            gauc = ev.auroc_conservative(log_conf_in.numpy(), ub.numpy())
            logged_scalars['test/' + out_dset + '_GAUC'] = gauc
            