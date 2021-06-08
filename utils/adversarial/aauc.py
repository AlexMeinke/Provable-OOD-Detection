import torch
import utils.adversarial.attacks as attacks
import utils.adversarial.apgd as apgd


def get_conf_lb(model, device, loader, args):
    restarts = 5
    budget = 5
    eps = args.eps
    num_classes = args.num_classes
    iterations = args.iterations
    from_logits = args.from_logits
    
    if num_classes==1:
        reduction = lambda x: torch.sigmoid(x)
    else:
        reduction = ( lambda x: torch.softmax(x, dim=1).max(1)[0] ) if from_logits else ( lambda x: x.exp().max(1)[0] )
    if 'use_last_class' in args and args['use_last_class']:
        use_last_class = True
        reduction = ( lambda x:  - torch.log_softmax(x, dim=1)[:,-1] )
        apgd_loss = 'last_conf'
        loss = attacks.LastConf()
    else:
        use_last_class = False
        apgd_loss = 'max_conf'
        loss = attacks.MaxConf(from_logits)
    
    attack = apgd.APGDAttack(model, n_iter=100*budget, n_iter_2=22*budget, n_iter_min=6*budget, size_decr=3,
                             norm='Linf', n_restarts=restarts, eps=eps, show_loss=False, seed=0,
                             loss=apgd_loss, show_acc=False, eot_iter=1, save_steps=False,
                             save_dir='./results/', thr_decr=.75, check_impr=False,
                             normalize_logits=False, device=device, apply_softmax=from_logits, classes=num_classes)
    
    stepsize = 0.1

    
    noise = attacks.DeContraster(eps)
    attack1 = attacks.MonotonePGD(eps, iterations, stepsize, num_classes, momentum=0.9, 
                                  norm='inf', loss=loss, normalize_grad=False, early_stopping=0, restarts=0,
                                  init_noise_generator=noise, model=model, save_trajectory=False)
    
    noise = attacks.UniformNoiseGenerator(min=-eps, max=eps)
    attack2 = attacks.MonotonePGD(eps, iterations, stepsize, num_classes, momentum=0.9, 
                                  norm='inf', loss=loss, normalize_grad=False, early_stopping=0, restarts=3,
                                  init_noise_generator=noise, model=model, save_trajectory=False)
    
    noise = attacks.NormalNoiseGenerator(sigma=1e-4)
    attack3 = attacks.MonotonePGD(eps, iterations, stepsize, num_classes, momentum=0.9, 
                                  norm='inf', loss=loss, normalize_grad=False, early_stopping=0, restarts=3,
                                  init_noise_generator=noise, model=model, save_trajectory=False)
    
    attack_list = [attack1, attack2, attack3]
    
    best = []
    for batch_idx, (x, y) in enumerate(loader):       
        if batch_idx==args.batches:
            break;
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_conf = reduction( model(x) ).detach().cpu()
        
        out = [ clean_conf ]
        if args.dataset!='RImgNet':
            attacked_point = model(attack.perturb(x.clone(), y)[0]).detach().cpu()
            out.append( reduction( attacked_point ) )
        
        for att in attack_list:
            attacked_point = att(x.clone(), y)
            o = model(attacked_point)
            o = reduction( o ).detach().cpu()
            out.append( o )
        
        max_conf, att_idx = torch.stack(out, dim=0).max(0)
        print(att_idx)
        
        best.append(max_conf)
    best = torch.cat(best, 0)
    return best
