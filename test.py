from deepface import DeepFace
path = "Brad_Pitt_Fury_2014.jpg"
import pandas as pd

models = ["dlib"]
#pred = DeepFace.analyze(img_path = path,  actions = ['emotion'] )
for model in models:
    # pred = DeepFace.detectFace("test/2.jpg", detector_backend=model, align=False)
    pred = DeepFace.extract_faces("Angelina-Jolie-Hollywood.jpg", detector_backend=model, align=False)
    print(model)
    print(pred[0]["confidence"])
    print("------------------------")

