from ultralytics import YOLO
import cv2 as cv



class DetectionModel:
    def __init__(self,path):
        self.model=YOLO(model=path)
        self.results=[]

    def run_inference(self,img):
        self.results=self.model(img,conf=0.4)

    def draw_box(self,img):
        mod_img=img.copy()

        for result in self.results:
            boxes=result.boxes
            for box in boxes:
                b=box.xyxy[0]

                x1,y1=int(b[0]),int(b[1])
                x2,y2=int(b[2]),int(b[3])

                cv.rectangle(mod_img,(x1,y1),(x2,y2),(0,255,0),1)

        return mod_img

    def predict(self,img):
        self.run_inference(img)
        mod_img=self.draw_box(img)
        return mod_img

if __name__=='__main__':
    model_path='detection_model.pt'

    model=DetectionModel(model_path)

    img=cv.imread('sample_1.jpg')

    mod_img=model.predict(img)

    cv.imshow('',mod_img)
    cv.waitKey(0)