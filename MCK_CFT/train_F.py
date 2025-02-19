import torch
from torch import nn
import torch.utils.data as Data
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import StepLR
import time
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
import warnings
warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sklearn.metrics import f1_score

def data_loader(data,args,drop_last = True, shuffle = True):
    generator=torch.Generator() 
    generator.manual_seed(args.seed) 

    loader = Data.DataLoader(
        dataset=data,
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=0,
        drop_last = drop_last,
        generator = generator,
    )
    return loader


def criterion(predictions, targets):
    return nn.NLLLoss()(input=predictions, target=targets)

def make_train_step(model, criterion, optimizer):
    def train_step(X, Y):
        output_gen, output_act = model(X)
        gen_predictions = torch.max(output_gen, 1)[1]
        act_predictions = torch.max(output_act, 1)[1]
        accuracy = torch.sum(Y[:,2] == act_predictions) / float(len(Y))
        loss = 0.8*criterion(output_gen, Y[:,0]) + 0.2*criterion(output_act, Y[:,2])
        loss = loss.to(device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy * 100 
    return train_step

def make_validate_fnc(model, criterion):
    def validate(test_dataloader,all_label):
        with torch.no_grad():
            model.eval()
            accuracy_act = 0
            accuracy_gen = 0
            loss = 0
            loop = tqdm((test_dataloader), total = len(test_dataloader))
            act_predictions = torch.empty([0]).to('cuda')
            gen_predictions = torch.empty([0]).to('cuda')
            Y_val_all = torch.empty([0]).to('cuda')
            validate_size = 0 
            Y_val_all
            for batch in loop:
                X_validate, Y_validate = batch
                actual_batch_size = X_validate.shape[0]
                validate_size += actual_batch_size
                X_validate = torch.tensor(X_validate, device=device).float()
                Y_validate = torch.tensor(Y_validate, dtype=torch.long, device=device) 
                output_gen, output_act = model(X_validate)
                predictions_act = torch.max(output_act, 1)[1]
                predictions_gen = []
                for pre_temps in predictions_act.tolist():
                    predictions_gen.append(int(all_label[pre_temps][0]))
                predictions_gen = torch.tensor(predictions_gen).to('cuda')                 
                Y_val_all = torch.cat([Y_val_all,Y_validate], dim=0)
                act_predictions = torch.cat([act_predictions, predictions_act], dim=0)
                gen_predictions = torch.cat([gen_predictions, predictions_gen], dim=0)
                accuracy_act += torch.sum(Y_validate[:,2] == predictions_act)
                accuracy_gen += torch.sum(Y_validate[:,0] == predictions_gen)
                loss += criterion(output_act, Y_validate[:,2]) * actual_batch_size 
            test_f1_score_macro = f1_score(Y_val_all[:,2].cpu(),act_predictions.cpu(), average='macro')
            accuracy_act = accuracy_act/validate_size
            accuracy_gen = accuracy_gen/validate_size
            loss = loss/validate_size
        return loss.item(), test_f1_score_macro * 100,accuracy_act * 100, act_predictions, gen_predictions,Y_val_all
    return validate


def make_save_checkpoint():
    def save_checkpoint(optimizer, model, epoch, filename):
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)
    return save_checkpoint


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch

def train(model, args,train_dataset,test_dataset,all_label):
    model_use_name = args.model
    num_epochs = args.epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
    train_step = make_train_step(model, criterion, optimizer=optimizer)
    validate = make_validate_fnc(model, criterion)
    save_checkpoint = make_save_checkpoint()
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        lr=optimizer.param_groups[0]['lr']
        train_dataloader = data_loader(train_dataset,args)
        test_dataloader = data_loader(test_dataset,args,drop_last = False, shuffle = False)
        model.train()
        epoch_acc = 0
        epoch_loss = 0
        train_size = 0
        step = len(train_dataloader)
        model.train()
        time_start = time.time()
        loop = tqdm((train_dataloader), total = len(train_dataloader))
        for batch in loop:
            X,Y = batch
            actual_batch_size = X.shape[0]
            X_tensor = torch.tensor(X, device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)            
            loss, acc = train_step(X_tensor, Y_tensor)
            train_size += actual_batch_size
            epoch_acc += acc * actual_batch_size 
            epoch_loss += loss * actual_batch_size 
        scheduler.step()
        epoch_acc = epoch_acc/train_size
        epoch_loss = epoch_loss/train_size
        valid_loss, valid_acc_act,valid_acc_gen, act_predictions, gen_predictions,Y_val_all = validate(test_dataloader,all_label)
        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)
        checkpoint_filename = '/root/autodl-tmp/model/model-{:03d}.pkl'.format(epoch)
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)
        save_output_txt =f'\nModel_use {model_use_name} ---Epoch {epoch} --- loss:{epoch_loss:.3f}, Epoch accuracy:{epoch_acc:.2f}%, test loss:{valid_loss:.3f}, test accuracy :{valid_acc_act:.2f}%'
        print(save_output_txt)