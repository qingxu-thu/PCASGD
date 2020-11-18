import torch
from torch.optim import Optimizer
from copy import deepcopy
from copy import copy
from collections import defaultdict



class PCASGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.95, dampening=0,
                 weight_decay=0, nesterov=False,**kwargs):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(PCASGD, self).__init__(params, defaults)
        #self.old_grad_groups = deepcopy(self.param_groups)
        self.agent_grad_groups= []
        self.predict_matrix = torch.zeros((self.n_agents))
        self.clip_matrix = torch.zeros((self.n_agents))
        self.pi = torch.FloatTensor(self.pi)
        for k in range(self.n_agents):
            if k != self.agent_id and self.relative_matrix[self.agent_id,k] > 1:
                self.predict_matrix[k] = 1
            if k != self.agent_id:
                if self.relative_matrix[self.agent_id,k] == 1:
                    self.clip_matrix[k] = 1



    def __setstate__(self, state):
        super(CDMSGD1, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, predict_start = False,closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        if not isinstance(self.state, defaultdict):
            self.state = defaultdict(dict)

        for i, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for j, p in enumerate(group['params']):

                #groups_use_para = [None]*self.n_agents
                #groups_use_grad = [None]*self.n_agents
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                m = 0
                param_state = self.state[p]
                if 'hb_buffer' not in param_state:
                    hb_buf = param_state['hb_buffer'] = torch.zeros(p.data.size()).cuda()
                else:
                    hb_buf = param_state['hb_buffer']

                hb_buf.mul_(momentum)
                hb_buf.add_(-group['lr'],d_p)
                
                #self.old_grad_groups[i]['params'][j].data = deepcopy(p.grad.data)
                #con_buf = torch.zeros(p.data.size()).cuda()
                con_buf = (p.data).mul_(self.pi[self.agent_id][self.agent_id])

                for k in range(self.n_agents):
                    if self.agent_param_groups[k] is not None:
                        if k==0:
                            groups_use_para = (self.agent_param_groups[k][i]['params'][j].data)
                        else:
                            groups_use_para = torch.cat((groups_use_para, self.agent_param_groups[k][i]['params'][j].data),0)
                    if self.agent_grad_groups[k] is not None:
                        if k==0:
                            groups_use_grad = (self.agent_grad_groups[k][i]['params'][j])
                        else:
                            groups_use_grad = torch.cat((groups_use_para, (self.agent_grad_groups[k][i]['params'][j]).data),0)
                    if k != self.agent_id:
                        if self.relative_matrix[self.agent_id,k] == 1:
                            m = m + 1
                    else:
                        self.pi[self.agent_id][k]=0
                (con_buf).add_(self.pi[self.agent_id], groups_use_para)       
                   
                '''
                for k in range(self.n_agents):
                    if self.agent_param_groups[k] is not None:
                        groups_use_para[k] = (self.agent_param_groups[k][i]['params'][j])
                    if self.agent_grad_groups[k] is not None:    
                        groups_use_grad[k] = (self.agent_grad_groups[k][i]['params'][j])
                    if k != self.agent_id:
                        if self.relative_matrix[self.agent_id,k] == 1:
                            m = m + 1
                '''        
                        #(con_buf).add_(self.pi[self.agent_id][k], groups_use_para[k])
                
                con_buf.add_(hb_buf)
                p.data = con_buf
                param_state['hb_buffer'] = hb_buf

                if predict_start:
                    predict_tensor = deepcopy(con_buf)
                    a = torch.add(p.data,groups_use_para[self.agent_id].mul_(-1))
                    predict_tensor.add_(groups_use_grad*groups_use_grad*a*-group['lr']*self.pi[self.agent_id]*self.predict_matrix)
                    
                    predict = torch.cosine_similarity(predict_tensor-p.data,d_p)

                    '''
                    for k in range(self.n_agents):
                        if k != self.agent_id and self.relative_matrix[self.agent_id,k] > 1:
                            temp_use = torch.zeros(p.data.size())
                            #predict_tensor.add_(torch.mul(groups_use_grad[k],-group['lr']*self.pi[self.agent_id][k]))
                            predict_tensor.add_(torch.mul(torch.mul(torch.mul(groups_use_grad[k],self.pi[self.agent_id][k]),torch.mul(groups_use_grad[k],torch.add(p.data,groups_use_para[self.agent_id].mul_(-1)))),-group['lr']))
                    '''
                    #predict = torch.norm(torch.mul(torch.add(predict_tensor,torch.mul(p.data,-1)),d_p))/torch.norm(torch.add(predict_tensor,torch.mul(p.data,-1)))
                    
                    clip_tensor = (p.data).mul_(1/(1+m))+hb_buf
                    (clip_tensor).add_(1/(1+m), groups_use_para*self.clip_matrix)
                    #clip = torch.norm(torch.mul(torch.add(clip_tensor,torch.mul(p.data,-1)),d_p))/torch.norm(torch.add(clip_tensor,torch.mul(p.data,-1)))
                    clip = torch.cosine_similarity(clip_tensor-p.data,d_p)
                    if predict > clip:
                        p.data = predict_tensor
                    else:
                        p.data = clip_tensor
                else:
                    p.data = con_buf
            return loss

    def set_agent_param_groups(self, agent_param_groups):
        self.agent_param_groups = deepcopy(agent_param_groups)

    def set_agent_grad_groups(self, agent_param_groups):
        self.agent_grad_groups = deepcopy(agent_param_groups)