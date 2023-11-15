import json
import numpy as np

import cv2
import os
from nltk.metrics.distance import jaccard_distance
import concurrent.futures
# construc

import datetime  
import re

from flask import render_template
import os
from flask import (Blueprint,
                   render_template,
                   redirect, url_for)

from flask import (Flask,
                   request,
                   redirect,
                   session,
                   send_file) 

                    
import pytesseract
import pymysql
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
# import requests
from PIL import Image
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
# Path of your tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
app = Flask(__name__, template_folder='template')



#------------------------------------------------------------------#




def tessaract():
    cheque= cv2.imread("static/result-img/Cheque No.jpg")
    account = cv2.imread("static/result-img/Account Holder Name.jpg")
    
    iban = cv2.imread("static/result-img/IBAN.jpg")
    logo = cv2.imread("static/result-img/logo.jpg")
    
    #convert the image to gray scale
    config = ('-l eng --oem 3 --psm 11')
    if cheque is not None:
        cheque_gray = cv2.cvtColor(cheque, cv2.COLOR_BGR2GRAY)
        cheque_text = pytesseract.image_to_string(cheque_gray, config=config)
    else:
        cheque_text= ""
    if account is not None:
        account_gray = cv2.cvtColor(account, cv2.COLOR_BGR2GRAY)
        account_text = pytesseract.image_to_string(account_gray, config=config)
    else:
        account_text= ""
    # amount_gray = cv2.cvtColor(amount, cv2.COLOR_BGR2GRAY)
    # date_gray = cv2.cvtColor(date, cv2.COLOR_BGR2GRAY)
    if iban is not None:
        iban_gray = cv2.cvtColor(iban, cv2.COLOR_BGR2GRAY)
        iban_text = pytesseract.image_to_string(iban_gray, config=config)
    else:
        iban_text= ""
    if logo is not None:
        logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        logo_text = pytesseract.image_to_string(logo_gray, config=config)
        #only take alphabets from logo_text
        logo_text = re.sub('[^A-Za-z]+', '', logo_text)
    else:
        logo_text= ""

    cheque_text = re.sub('[^0-9]', '', cheque_text)
    with open('sample.txt', 'r') as f:
        bank_name =f.read().split()
    for i in bank_name:
        bank= 1 - jaccard_distance(set(logo_text), set(i))
        print(bank)
        if bank > 0.3:
            logo_text = i
            print(bank)
            print(logo_text)
    return cheque_text, account_text, iban_text, logo_text



#------------------------------------------------------------------#




def pay():
    img_1 = Image.open("static/result-img/Pay.jpg").convert("RGB")
    pixel_values_1 = processor(img_1, return_tensors="pt").pixel_values
    generated_ids_1 = model.generate(pixel_values_1)
    generated_text_1 = processor.batch_decode(generated_ids_1, skip_special_tokens=True)[0]
    return generated_text_1



#------------------------------------------------------------------#




def date():
    img_2 = Image.open("static/result-img/date.jpg").convert("RGB")
    pixel_values_2 = processor(img_2, return_tensors="pt").pixel_values
    generated_ids_2 = model.generate(pixel_values_2)
    generated_text_2 = processor.batch_decode(generated_ids_2, skip_special_tokens=True)[0]
    #only digit will be extracted from the date
    date_text = re.sub('[^0-9]', '', generated_text_2)
    #spilit the date into day, month and year
    day = date_text[0:2]
    month = date_text[2:4]
    year = date_text[4:8]
    #convert the date into dd/mm/yyyy format
    date_text = day + "-" + month + "-" + year
    return date_text



#------------------------------------------------------------------#




def amount():
    img_3= Image.open("static/result-img/Amount.jpg").convert("RGB")
    pixel_values_3 = processor(img_3, return_tensors="pt").pixel_values
    generated_ids_3 = model.generate(pixel_values_3)
    generated_text_3 = processor.batch_decode(generated_ids_3, skip_special_tokens=True)[0]
    #only digit will be extracted from the amount
    amount_text = re.sub('[^0-9]', '', generated_text_3)
    return amount_text



#------------------------------------------------------------------#




@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')



#------------------------------------------------------------------#




@app.route('/text', methods=['GET', 'POST'])
def text():
    if request.method=='POST':
        conn = pymysql.connect(db='cheque', user='root', passwd='', host='localhost')
        cur = conn.cursor()

        ct = str("static/result/")+str("Signature")+str(datetime.datetime.now())+str(".jpg")
        account_name = request.form['account_name']
        amount = request.form['amount']
        cheque_no = request.form['cheque_no']
        date = request.form['date']
        iban = request.form['iban']
        bank_name = request.form['bank_name']
        pay= request.form['pay']
        # print(account_name,amount,cheque_no,date,iban,bank_name,pay)
        cur.execute("INSERT INTO cheque_classfication (account_name,amount,cheque_no,date,iban,bank_name,pay,signature) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",(account_name,amount,cheque_no,date,iban,bank_name,pay,ct))
        conn.commit()
        cur.close()
        conn.close()


    return render_template('text.html')




#------------------------------------------------------------------#




@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # call the main function from file.py
    if request.method == 'POST':
        f = request.files['file1']
        #save the file with different name
        #delete all the file in the folder
        for filename in os.listdir('static/upload-img'):
            os.remove('static/upload-img/'+filename)
        for filename in os.listdir('static/result-img'):
            os.remove('static/result-img/'+filename)
        f.save(os.path.join('static/upload-img/', f.filename))
        # return render_template('index.html', filename=f.filename)
        # pass the image to the main function
        labelsPath = "yolo/obj.names"
        LABELS = open(labelsPath).read().strip().split("\n")
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")

        weightsPath ="yolo/yolov4-custom_best.weights"
        configPath = "yolo/yolov4-custom.cfg"


        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        im=[]
        # read the image
        frame = cv2.imread("static/upload-img/"+f.filename)
        
        Num = 0
        
        (H, W) = frame.shape[:2]

        
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []
        confidence_threshold = 0.5
        overlapping_threshold = 0.6

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidence_threshold:
                    # Scale the bboxes back to the original image size
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Remove overlapping bounding boxes and boundig boxes
        bboxes = cv2.dnn.NMSBoxes(
            boxes, confidences, confidence_threshold, overlapping_threshold)
        if len(bboxes) > 0:
            for i in bboxes.flatten():
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
              
                
                crop_image = frame[y:y+h, x:x+w]

                
              
                
                try:
                    if LABELS[classIDs[i]] == "A-C name":
                        
                        cv2.imwrite("static/result-img/"+"Account Holder name"+".jpg", crop_image)
                        # continue
                    elif LABELS[classIDs[i]] == "Amount":
                        #convert the image to gray scale
                        # crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite("static/result-img/"+"Amount"+".jpg", crop_image)
                        # continue
                    elif LABELS[classIDs[i]] == "Logo":
                        cv2.imwrite("static/result-img/"+"logo"+".jpg", crop_image)
                        # continue
                    elif LABELS[classIDs[i]] == "Cheque No":
                        cv2.imwrite("static/result-img/"+"Cheque No"+".jpg", crop_image)
                        # continue
                    elif LABELS[classIDs[i]] == "Date":
                        # crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite("static/result-img/"+"Date"+".jpg", crop_image)
                        # continue
                    elif LABELS[classIDs[i]] == "IBAN":
                        cv2.imwrite("static/result-img/"+"IBAN"+".jpg", crop_image)
                        # continue
                    elif LABELS[classIDs[i]] == "Signature":
                        ct = datetime.datetime.now()
                        cv2.imwrite("static/result-img/"+"Signature"+".jpg", crop_image)
                        
                        # continue
                    elif LABELS[classIDs[i]] == "Pay":
                        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite("static/result-img/"+"Pay"+".jpg",crop_image)
                    elif (LABELS[classIDs[i]]) == "MICR":
                        cv2.imwrite("static/result-img/"+"MICR"+".jpg",frame[y:y+h, x:x+w])
                    #     # continue
                        # continue
                    
                    else:
                        continue
                except:
                    print("Sorry! NO FRAME ACCESSED!")
                
                Num = Num + 1
        # extraction() 
        
        cheque= cv2.imread("static/result-img/Cheque No.jpg")
        account = cv2.imread("static/result-img/Account Holder Name.jpg")
        # amount = cv2.imread("static/result-img/Amount.jpg")
        # date = cv2.imread("static/result-img/Date.jpg")
        iban = cv2.imread("static/result-img/IBAN.jpg")
        logo = cv2.imread("static/result-img/logo.jpg")
        # pay= cv2.imread("static/result-img/Pay.jpg")
        #preprocess the image
        #convert the image to gray scale
        config = ('-l eng --oem 3 --psm 11')
        if cheque is not None:
            cheque_gray = cv2.cvtColor(cheque, cv2.COLOR_BGR2GRAY)
            cheque_text = pytesseract.image_to_string(cheque_gray, config=config)
        else:
            cheque_text= ""
        if account is not None:
            account_gray = cv2.cvtColor(account, cv2.COLOR_BGR2GRAY)
            account_text = pytesseract.image_to_string(account_gray, config=config)
        else:
            account_text= ""
        # amount_gray = cv2.cvtColor(amount, cv2.COLOR_BGR2GRAY)
        # date_gray = cv2.cvtColor(date, cv2.COLOR_BGR2GRAY)
        if iban is not None:
            iban_gray = cv2.cvtColor(iban, cv2.COLOR_BGR2GRAY)
            iban_text = pytesseract.image_to_string(iban_gray, config=config)
        else:
            iban_text= ""
        if logo is not None:
            logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
            logo_text = pytesseract.image_to_string(logo_gray, config=config)
            #only take alphabets from logo_text
            logo_text = re.sub('[^A-Za-z]+', '', logo_text)
        else:
            logo_text= ""


        cheque_text = re.sub('[^0-9]', '', cheque_text)
        with open('sample.txt', 'r') as f:
            bank_name =f.read().split()
        for i in bank_name:
            bank= 1 - jaccard_distance(set(logo_text), set(i))
            print(bank)
            if bank > 0.3:
                logo_text = i
                print(bank)
                print(logo_text)
                
        #run the extraction function on a thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(date)
                future_1 = executor.submit(pay)
                future_2 = executor.submit(amount)
                future_3 = executor.submit(tessaract)
                return_value = future_3.result()
                cheque_text, account_text, iban_text, logo_text=return_value
                date_text = future.result()
                pay_text = future_1.result()
                amount_text = future_2.result()
            
        return render_template('text.html',cheque_text=cheque_text, account_text=account_text, amount_text=amount_text, date_text=date_text, iban_text=iban_text, logo_text=logo_text, pay_text=pay_text)
       





#------------------------------------------------------------------#





if __name__ == '__main__':
   
    app.run(debug=True)
