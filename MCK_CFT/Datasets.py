import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data as Data



def re_name_label(all_labels_list):
    uniques_gen = np.unique(all_labels_list)
    for k in range(len(uniques_gen)):
        gen_label_temp = np.zeros(len(np.where(all_labels_list == uniques_gen[k])[0]))+k
        if k == 0:
            gen_label = gen_label_temp

        else:
            gen_label = np.concatenate((gen_label, gen_label_temp), axis = 0)
    return gen_label  


def re_name_filenames(filenames):
    all_labels_filenames =[]
    file_names_new = np.copy(filenames)
    for k in range(len(filenames)):
        all_labels_filenames_temp = np.vstack((( int(filenames[k][3:5]) - 1),(int(filenames[k][6:8]) - 1), (k)))
        all_labels_filenames_temp = np.transpose(all_labels_filenames_temp)
        if k == 0:
            all_labels_filenames = all_labels_filenames_temp

        else:
            all_labels_filenames = np.concatenate((all_labels_filenames, all_labels_filenames_temp), axis = 0)

    gen_label = re_name_label(all_labels_filenames[:,0])
    spe_label = re_name_label(all_labels_filenames[:,1])
    act_label = re_name_label(all_labels_filenames[:,2])
    all_label = np.transpose(np.vstack((gen_label, spe_label,act_label)))
    for k in range(len(filenames)):
        list_filename_temp = list(filenames[k])
        list_filename_temp[3:5] = "%02d" % (all_label[k,0])
        list_filename_temp[6:8] = "%02d" % (all_label[k,1])
        list_filename_temp[0:2] = "%02d" % (all_label[k,2])
        file_names_new[k]=''.join(list_filename_temp)         
    return file_names_new,all_label.astype(int)


def min_max_nor(value):
    new_value = ((value - value.min()) / (value.max() - value.min()))
    return new_value

def Fusion_prod(data_shape,train_dataset_raman,train_dataset_mass,raman_num_to):
    Fusion_Data = np.empty((data_shape, raman_num_to, raman_num_to))
    for k in range(data_shape):
        aaa = train_dataset_raman[k][:,np.newaxis]
        bbb = train_dataset_mass[k][np.newaxis,:]
        Fusion_Data[k] = np.matmul(aaa,bbb)
    Fusion_Data = Fusion_Data.astype(np.float32)
    return Fusion_Data

def mass_enhance(dataset_raman,labels_r,dataset_mass,labels_mo):
    r_label_series = pd.Series(labels_r[:,2])
    r_class_count = r_label_series.value_counts().sort_index()    
    mo_label_series = pd.Series(labels_mo[:,2])
    mo_class_count = mo_label_series.value_counts().sort_index()    
    class_indices_range = pd.DataFrame({
        'labels': labels_mo[:,2],
        'indices': range(len(labels_mo[:,2]))
    }).groupby('labels')['indices'].agg(['min', 'max']).reset_index()
    dataset_mass_output = None
    for i in range(r_class_count.shape[0]):   
        raman_num = r_class_count[i]
        mass_num = mo_class_count[i]
        yu_r_m = raman_num % mass_num 
        fold_r_m = raman_num // mass_num  
        x_m_w_temp = np.tile(dataset_mass[class_indices_range['min'][i]:class_indices_range['max'][i]+1], (fold_r_m, 1)) 
        x_m_w_temp = np.vstack((x_m_w_temp, dataset_mass[class_indices_range['min'][i]:class_indices_range['max'][i]+1][0:yu_r_m,:]))
        dataset_mass_pre = np.empty((x_m_w_temp.shape[0], x_m_w_temp.shape[1]))
        np.random.seed(100)
        for k in range(x_m_w_temp.shape[0]):
            r = np.random.normal(loc=0.0, scale=0.3, size=(x_m_w_temp.shape[1], 1))
            for z in range(x_m_w_temp.shape[1]):
                dataset_mass_pre[k, z] = (x_m_w_temp[k, z]) * (1 + r[z, 0])
        dataset_mass_pre = dataset_mass_pre.astype(np.float32)
        dataset_mass_output = dataset_mass_pre if dataset_mass_output is None else np.concatenate((dataset_mass_output, dataset_mass_pre), axis=0)
    return dataset_mass_output
def Load_labels():
    filenames = ['01_01_01_M.Huston','02_01_02_M.abscessus','03_01_03_M.phlei','04_01_04_M.mucogenicum','05_01_05_M.ulcerans',
             '06_01_06_M.peregrinum','07_01_07_M.chelonae','08_01_08_M.smegmatis','09_01_09_M.sauton','10_01_10_M.gordonae']
    file_num = len(filenames)
    file_list = [*range(0, len(filenames), 1)]
    action_dict = dict(zip(file_list, filenames))
    file_names_new,all_label = re_name_filenames(filenames)    
    return action_dict,all_label

def Load_dataset(path):
    save_name_train = path + '/train.pt'
    loaded_train_data = torch.load(save_name_train)  
    train_dataset_r = loaded_train_data['train_dataset_r']  
    train_dataset_mo = loaded_train_data['train_dataset_mo']  
    train_labels_r = loaded_train_data['train_labels_r']  
    train_labels_mo = loaded_train_data['train_labels_mo']  
    save_name_test = path + '/test.pt'
    loaded_test_data = torch.load(save_name_test)  
    test_dataset_r = loaded_test_data['test_dataset_r']  
    test_dataset_mo = loaded_test_data['test_dataset_mo']  
    test_labels_r = loaded_test_data['test_labels_r']  
    test_labels_mo = loaded_test_data['test_labels_mo']  
    train_dataset_mass = mass_enhance(train_dataset_r,train_labels_r,train_dataset_mo,train_labels_mo)
    test_dataset_mass = mass_enhance(test_dataset_r,test_labels_r,test_dataset_mo,test_labels_mo)
    return train_dataset_r,test_dataset_r,train_dataset_mass,test_dataset_mass,train_labels_r,test_labels_r
    
def Fusion_Data(data_path):
    action_dict,all_label = Load_labels()
    train_raman,test_raman,train_mass,test_mass,train_y,test_y = Load_dataset(data_path)
    FData_train = Fusion_prod(train_raman.shape[0],train_raman,train_mass,train_raman.shape[1])
    FData_test = Fusion_prod(test_raman.shape[0],test_raman,test_mass,train_raman.shape[1])   
    train_dataset = Data.TensorDataset(torch.tensor(FData_train).unsqueeze(1), torch.tensor(train_y))
    test_dataset = Data.TensorDataset(torch.tensor(FData_test).unsqueeze(1), torch.tensor(test_y))
    return train_dataset,test_dataset,action_dict,all_label

def Linear_Data(data_path):
    action_dict,all_label = Load_labels()
    train_raman,test_raman,train_mass,test_mass,train_y,test_y = Load_dataset(data_path)
    for k in range(train_raman.shape[0]):
        train_raman[k] = min_max_nor(train_raman[k])
    for k in range(test_raman.shape[0]):
        test_raman[k] = min_max_nor(test_raman[k])
    LData_train = torch.cat([torch.tensor(train_raman).unsqueeze(1), torch.tensor(train_mass).unsqueeze(1)],dim = 2)
    LData_test = torch.cat([torch.tensor(test_raman).unsqueeze(1), torch.tensor(test_mass).unsqueeze(1)],dim = 2)
    train_dataset = Data.TensorDataset(torch.tensor(LData_train), torch.tensor(train_y))
    test_dataset = Data.TensorDataset(torch.tensor(LData_test), torch.tensor(test_y))
    return train_dataset,test_dataset,action_dict,all_label


