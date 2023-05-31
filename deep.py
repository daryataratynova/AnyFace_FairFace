from deepface import DeepFace
import argparse
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages
import os
import csv



def detect(
    img_size,
    conf_thres,
    data,
    model_name
):
    path = "fairface/" + data + "/" #e.g. fairface/val
    dataset = LoadImages(path, img_size) #load images from folder mentioned above
    for path, im, im0s, vid_cap in dataset: #for each image
        try: #try to get a face
            pred = DeepFace.extract_faces(path, detector_backend=model_name, enforce_detection= True)
            #if we got a face then we check for confidence
            if pred[0]['confidence']> conf_thres: 
                found = True #if confidence is above conf_thres then we set status to True
            else:
                found = False #if condidence is below then we set status to False
        except: #if we cannot get a face we set status to Fale
                found = False

        if (found == False): #if model could not find a face then we save path
            if data == "train":
                    file  = "fairface/"+ model_name+"/conf_thres="+ str(conf_thres) + "/train_undetected.csv"
                    file_exists = os.path.isfile(file)
            else:
                    file  = "fairface/" + model_name + "/conf_thres="+ str(conf_thres) + "/val_undetected.csv"
                    file_exists = os.path.isfile(file)

            with open (file, 'a') as csvfile:
                headers = ['file']
                writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({'file': path})

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixesls)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='confidence thr')
    parser.add_argument('--data', type =str, default = 'val', help = 'train or val')
    parser.add_argument('--model_name', type=str, default = 'opencv', help = 'model name')
    opt = parser.parse_args()

    detect(opt.img_size, opt.conf_thres, opt.data, opt.model_name)