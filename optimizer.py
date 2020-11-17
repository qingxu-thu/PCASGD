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
		super(CDMSGD1, self).__init__(params, defaults)
		self.old_param_groups = deepcopy(self.param_groups)
		self.old_grad_groups = deepcopy(self.param_groups)
		self.agent_grad_groups= []

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
				groups_use_para = [None]*self.n_agents
				groups_use_grad = [None]*self.n_agents
				if p.grad is None:
					continue
				d_p = deepcopy(p.grad.data)
			
				m=0	
				if torch.norm(p,1)<50 and torch.norm(d_p,1)>200: 
					print("grad",d_p)
					time.sleep(60)

				param_state = self.state[p]
				if 'hb_buffer' not in param_state:
					hb_buf = param_state['hb_buffer'] = torch.zeros(p.data.size()).cuda()
				else:
					hb_buf = param_state['hb_buffer']
				#hb_buf = hb_buf.to(self.agent_id)
				hb_buf.mul_(momentum)
				hb_buf.add_(-group['lr'],d_p)

				self.old_grad_groups[i]['params'][j].data = deepcopy(p.grad.data)
				con_buf = torch.zeros(p.data.size()).cuda()
				con_buf.add_(p.data).mul_(self.pi[self.agent_id][self.agent_id])
				for k in range(self.n_agents):
					#print(k)
					if self.agent_param_groups[k] is not None:
						groups_use_para[k] = (self.agent_param_groups[k][i]['params'][j].data)
					if self.agent_grad_groups[k] is not None:	
						groups_use_grad[k] = (self.agent_grad_groups[k][i]['params'][j].data)
					if k != self.agent_id:
						if self.relative_matrix[self.agent_id,k] == 1:
							m = m + 1
						(con_buf).add_(self.pi[self.agent_id][k], groups_use_para[k])


				con_buf.add_(hb_buf)
				p.data = con_buf
				param_state['hb_buffer'] = hb_buf
				if predict_start:
					predict_tensor = deepcopy(con_buf)
					for k in range(self.n_agents):
						if k != self.agent_id and self.relative_matrix[self.agent_id,k] > 1:
							temp_use = torch.zeros(p.data.size())
							#predict_tensor.add_(torch.mul(groups_use_grad[k],-group['lr']*self.pi[self.agent_id][k]))
							predict_tensor.add_(torch.mul(torch.mul(torch.mul(groups_use_grad[k],self.pi[self.agent_id][k]),torch.mul(groups_use_grad[k],torch.add(p.data,groups_use_para[self.agent_id].mul_(-1)))),-group['lr']))
					predict =  torch.norm(torch.mul(torch.add(predict_tensor,torch.mul(p.data,-1)),d_p))/torch.norm(torch.add(predict_tensor,torch.mul(p.data,-1)))
					clip_tensor = torch.zeros(p.data.size()).cuda()
					clip_tensor.add_(p.data).mul_(1/(1+m))
					clip_tensor.add_(hb_buf)
					#print(m)
					for k in range(self.n_agents):
							if k != self.agent_id:
								if self.relative_matrix[self.agent_id,k] == 1:
									(clip_tensor).add_(1/(1+m), groups_use_para[k])
					clip =  torch.norm(torch.mul(torch.add(clip_tensor,torch.mul(p.data,-1)),d_p))/torch.norm(torch.add(clip_tensor,torch.mul(p.data,-1)))
					if predict > clip:
					#	print("p")
						p.data = predict_tensor
					else:
					#	print("c")
						p.data = clip_tensor
					#if torch.norm(clip_tensor)>100000:
					#	print("error")
				else:
					p.data = con_buf
			return loss

	def set_agent_param_groups(self, agent_param_groups):
		self.agent_param_groups = deepcopy(agent_param_groups)

	def set_agent_grad_groups(self, agent_param_groups):
		self.agent_grad_groups = deepcopy(agent_param_groups)