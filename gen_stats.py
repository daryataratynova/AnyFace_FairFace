import pandas as pd
import argparse

#returns percentage of each group in undetected
def ratio_in_undetected(data, col):
    col_classes = data.groupby(col)[col].count()
    return round(col_classes/data[col].count()*100, 2)

#returns percentage of each group in total
def ratio_in_total(data, total_data, col):
     col_classes = data.groupby(col)[col].count()
     return round(col_classes/total_data.groupby(col)[col].count()*100, 2)

#returns ratio of undetected/total
def ratio(data, total_data):
     return round(data.shape[0]/total_data.shape[0]*100, 2)

def create_table(col, undetected_set, total_set, total, file):
     for i in col:
          if (total == True) :
               res = ratio_in_total(undetected_set, total_set, i).rename()
          else:
               res = ratio_in_undetected(undetected_set, i).rename()
          df = res.to_frame().reset_index()
          df.columns = [i, '%']
          file.write(df.to_markdown())
          file.write("\n\n\n")

#generates markdown file for one model
def gen_markdown(conf_thres, model,  ):

     undetected_train, train, undetected_val, val = read_files(conf_thres, model)
     file = open("fairface/"+ model +"/conf_thres=" + str(conf_thres) + "/balanced/stats_balanced.md", "a+") # create markdown file for stats
     #Making Title
     file.write("# Results for confidence threshold {} \n".format(conf_thres))
     file.write("## On train dataset \n")
     ratio_undetected_train = ratio(undetected_train, train)
     file.write("Undetected percentage is {} % \n".format(ratio_undetected_train))
     file.write("### Proportion in Undetected by Group \n")
     create_table(columns, undetected_train, train, False, file)
     file.write("### Proportion of in Total by Group \n")
     create_table(columns, undetected_train, train, True, file)
     file.write("## On val dataset\n")
     ratio_undetected_val = ratio(undetected_val, val)
     file.write("Undetected percentage is {} % \n".format(ratio_undetected_val))
     file.write("### Proportion in Undetected by Group \n")
     create_table(columns, undetected_val, val, False, file)
     file.write("### Proportion in Total by Group \n")
     create_table(columns, undetected_val, val, True, file)

def read_files(conf_thres, model):
     train = pd.read_csv("fairface/fairface_label_train.csv")
     val = pd.read_csv("fairface/fairface_label_val.csv")

     #choose rows with service_test = True to get balanced dataset
     train = train[(train['service_test'] == True)] 
     val = val[(val['service_test'] == True)] 

     #read undetected faces 
     undetected_train = pd.read_csv("fairface/"+ model +"/conf_thres=" + str(conf_thres) + "/balanced/train_balanced.csv")
     undetected_val = pd.read_csv("fairface/"+ model +"/conf_thres=" + str(conf_thres) + "/balanced/val_balanced.csv") 

     return undetected_train, train, undetected_val, val

#generate csv file
def gen_csv(param, conf_thres, models):
     result_train, result_val = pd.DataFrame(), pd.DataFrame()
     undetected_percentage_train = []
     undetected_percentage_val = []
     i = 0
     for model in models:
          undetected_train, train, undetected_val, val = read_files(conf_thres, model)
          res_train = ratio_in_undetected(undetected_train, param).rename().to_frame().reset_index()
          res_train.columns = [param, '%']
          result_train.insert(i, model, res_train["%"])
          undetected_percentage_train.append(ratio(undetected_train, train))
          res_val = ratio_in_undetected(undetected_val, param).rename().to_frame().reset_index()
          res_val.columns = [param, '%']
          result_val.insert(i, model, res_val["%"])
          undetected_percentage_val.append(ratio(undetected_val, val))
          i+=1
     result_train.loc[len(result_train)] = undetected_percentage_train
     result_train.to_csv("results_csv/" + param + "_train.csv", i)
     result_val.loc[len(result_val)] = undetected_percentage_val
     result_val.to_csv("results_csv/" + param + "_val.csv")


if __name__ == '__main__':
     columns = ["age", "gender", "race"]
     models = ["AnyFace", "Yolov5l", "retinaface", "opencv", "ssd", "mtcnn", "dlib"] #, "Yolov5l", "retinaface", "opencv", "ssd", "mtcnn", "dlib"

     parser = argparse.ArgumentParser()
     parser.add_argument('--conf_thres', type=float, default = 0.5, help='confidence thr')
     parser.add_argument('--model_name', type=str, default = 'retinaface', help = 'model name')
     parser.add_argument('--stats_format', type=str, default = 'csv', help = 'model name')
     parser.add_argument('--param', type = str, default = 'gender', help = 'age, gender or race')
     opt = parser.parse_args()
     
     if opt.stats_format == 'markdown':
          gen_markdown(opt.conf_thres, opt.model_name)
     elif opt.stats_format == 'csv': 
          gen_csv(opt.param, opt.conf_thres, models)
