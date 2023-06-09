import pandas as pd
import argparse

def merge(conf_thres, model_name):
    #get csv with path to undetected faces
    train_undetected = pd.read_csv("fairface/"+ model_name +"/conf_thres=" + str(conf_thres)+ "/train_undetected.csv")
    val_undetected = pd.read_csv("fairface/"+ model_name +"/conf_thres=" + str(conf_thres)+ "/val_undetected.csv")
    
    #delete /fairface
    train_undetected["file"] = train_undetected["file"].map(lambda x: x.lstrip("fairface/").rstrip('aAbBcC'))
    val_undetected["file"] = val_undetected["file"].map(lambda x: x.lstrip("fairface/").rstrip('aAbBcC'))

    #get csv with path to faces                           
    train_set = pd.read_csv("fairface/fairface_label_train.csv")
    val_set = pd.read_csv("fairface/fairface_label_val.csv")

    #choose rows with service_test = True to get balanced dataset
    train_set = train_set[(train_set['service_test'] == True)]
    val_set = val_set[val_set['service_test'] == True]

    #merge undetected and balanced set s.t. we get undetected balanced set with all columns
    train_merged = pd.merge(train_undetected, train_set, on = "file")
    val_merged = pd.merge(val_undetected,val_set, on = "file")

    #save undetected balanced set to balanced folder
    train_merged.to_csv("fairface/"+ model_name +"/conf_thres="+str(conf_thres)+"/balanced/train_balanced.csv")
    val_merged.to_csv("fairface/"+ model_name +"/conf_thres="+str(conf_thres)+"/balanced/val_balanced.csv")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_thres', type=float, default = 0.5, help='confidence thr')
    parser.add_argument('--model_name', type=str, default = 'opencv', help = 'model name')
    opt = parser.parse_args()
    

    merge(opt.conf_thres, opt.model_name)