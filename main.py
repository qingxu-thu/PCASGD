import numpy as np




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