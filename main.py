import numpy as np
import torch
from optimizer import PCASGD


def optimizer_inital(num_agents,model_list,kwargs):
    optimizer_list = []
    for itr,model in enumerate(model_list):
        optimizer = PCASGD(model.parameters(), lr=kwargs['learning_rate'], momentum=kwargs['momentum'])
        optimizer.pi = pi
        optimizer.n_agents = kwargs['n_agents']
        optimizer.agent_id = itr
        optimizer.relative_matrix = relative_matrix


def train(model_list, train_data_list,optimizer_list ,epoch,criterion,is_cuda=True):
    for model in model_list:
        model.train()
    total_train_loss = 0.
    for u in range(len(model_list)):
        for batch_idx, (inputs, target) in enumerate(train_data_list[u]):
            if is_cuda:
                inputs, target = inputs.cuda(), target.cuda()
            #inputs, target = Variable(inputs), Variable(target)
            optimizer_list[u].zero_grad()
            train_loss = 0.
            outputs = model_list[u](inputs)
            train_loss = criterion(outputs, target)
            train_loss.backward()
            if epoch > 4:
                optimizer_list[u].step(True)
            else:
                optimizer_list[u].step()    
            total_train_loss += train_loss.data[0]
        epoch_loss = total_train_loss/len(self.train_data_loader)
        self.train_loss_hist.append(epoch_loss.item())
        return epoch_loss



def share_params(self, epoch, relation_matrix, optimizer_list):
    epoch = int(epoch)
    delay = int(np.max(relation_matrix))
    if epoch>delay:
        for itr, optimizer in optimizer_list:
            agent_param_groups = [None]*self.n_agents
            agent_grad_groups = [None]*self.n_agents
                        
        for agent in self.agents:
            agent_param_groups = [None]*self.n_agents
            agent_grad_groups = [None]*self.n_agents
            for i in range(self.n_agents):
                num = agent.clock(self.relation_matrix[agent.agent_id,i],self.delay,epoch)
                agent_param_groups[i] = self.param_pool[num]
                agent_grad_groups[i] = self.grad_pool[num]
        agent.set_params_groups(agent_param_groups)
        agent.set_agent_grad_groups(agent_grad_groups)

        agent_state_dicts = [agent.get_states() for agent in self.agents]
        for agent in self.agents:
            agent_states = []
            agent_param_groups = []
            for state_dict in agent_state_dicts:

                if not isinstance(state_dict['state'], defaultdict):
                    agent_state = defaultdict(dict)
                else:
                    agent_state = state_dict['state']
                agent_states.append(agent_state)
                self.param_pool[self.delay*agent.agent_id+epoch%self.delay]=state_dict['param_groups']
                #agent_param_groups.append(state_dict['param_groups'])
                self.grad_pool[self.delay*agent.agent_id+epoch%self.delay]=agent.get_grad_groups()
    else:
        agent_state_dicts = [agent.get_states() for agent in self.agents]
        for agent in self.agents:
            agent_states = []
            agent_param_groups = []
            agent_grad_groups = []
            for state_dict in agent_state_dicts:
                if not isinstance(state_dict['state'], defaultdict):
                    agent_state = defaultdict(dict)
                else:
                    agent_state = state_dict['state']
                agent_states.append(agent_state)
                self.param_pool[self.delay*agent.agent_id+epoch%self.delay]=state_dict['param_groups']
                agent_param_groups.append(state_dict['param_groups'])
                self.grad_pool[self.delay*agent.agent_id+epoch%self.delay]=agent.get_grad_groups()
                agent_grad_groups.append(agent.get_grad_groups())
                agent.set_params_groups(agent_param_groups)
                agent.set_agent_grad_groups(agent_grad_groups)