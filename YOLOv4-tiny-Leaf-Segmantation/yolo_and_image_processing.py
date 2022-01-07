# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 23:45:58 2021

@author: Cinar
"""

#opencv versıon komtrol

import cv2
print(cv2.__version__)


#%% 1. Bölüm

"""
Burada kütüphanelerimizi import ediyoruz
Görüntü işleme için OPenCV
Matris işlemleri için numpy ve Pandas
Görselleştirme için matplotlib
"""

import cv2
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




image = cv2.imread("test (2).jpg")
# print(img)

img_width = image.shape[1]
img_height = image.shape[0]
plt.imshow(image)
plt.axis("off") 
plt.title("Original Image")




#%% Gray Scale

"""
Burada resmimizi renkli formattan gri formata dönüştürüyoruz
ve görselleştiriyoruz.

"""
# Gray scale Coversion
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(image_gray)
plt.axis("off") 
plt.title("Gray")
# cv2.imshow('Yaprak',image_gray)
# cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows()


#%% Gaussian Blur

""" # Gaussian Blur """

"""
Burada resmimize bluring işlemi yapıyoruz ve bulanıklaştırıyoruz
ve görselleştiriyoruz.

"""
image_gb = cv2.GaussianBlur(image_gray, ksize = (3,3), sigmaX = 7)
plt.figure(figsize=(10,6))
plt.imshow(image_gb)
plt.axis("off")
plt.title("Gaussian Blur")


#%% Thresholding

""" # Thresholding """


"""
Burada resmimize eşikleme yapıyoruz renk skalasın 100 ün altında olan piksellerimizi beyaz,
geriye kalanlarını ise siyah yapıyoruz.
Özet geçersek verilen resmi ikili binary görüntüye cevirmek .Siyah-Beyaz 

"""

_, thresh_img = cv2.threshold(image_gb, thresh = 100, maxval = 255, type = cv2.THRESH_BINARY)



plt.figure()
thresh_img = cv2.bitwise_not(thresh_img, mask = None)
# thresh_img = cv2.resize(thresh_img, (227,227), interpolation = cv2.INTER_AREA)
plt.imshow(thresh_img, cmap = "gray")
plt.title("Thresholding")

# print(thresh_img.shape)


# %% #Generaing blank black image of same dimension 

"""
Burada blank oluştırmak için bir tane fonksiyon tanımlıyoruz
bu fonksiyon rengini istegimize göre ayarlayabilecegimiz SİYAH blank olusturacak.


"""
def create_blank(width, height, rgb_color=(0, 0, 0)):
    
    """
    Create new image(numpy array) filled with certain color in RGB
    RGB'de belirli bir renkle doldurulmuş yeni görüntü (numpy dizisi) oluşturun.
    """
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


# Create new blank 300x300 Black image

"""
Burada yeni black blankımıza hoşgeldin diyebiliriz :)
    
"""
width1, height1 = image.shape[0], image.shape[1]

black = (0, 0, 0)
thresh_img2 = create_blank(width1, height1, rgb_color=black)
plt.figure() 
plt.imshow(thresh_img2)
plt.title("New Black Blank")


# %% # contours 

""" # contours """

"""
Burada aynı renk ve yoğunluğa sahip olan kesintisiz noktaları sınır boyunca
birleştiren bir eğri oluşturuyoruz.

"""
_,contours, hierarch = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

external_contour = np.zeros(thresh_img.shape)
internal_contour = np.zeros(thresh_img.shape)

for i in range(len(contours)):
    
    # external
    if hierarch[0][i][3] == -1:
        cv2.drawContours(external_contour,contours, i, 255, -1)
    else: # internal
        cv2.drawContours(internal_contour,contours, i, 255, -1)

plt.figure() 
plt.imshow(external_contour, cmap = "gray"),
plt.axis("off")
plt.title("External Contours")
#plt.title("Internal Contours")


"""
#plt.figure(), plt.imshow(internal_contour, cmap = "gray"),plt.axis("off")
Eğer içten kontür çizdirmek istiyorsanız bunu kullanın

""" 

# %% Bitwise Masking & FINAL

""" Bitwise Masking  """
"""
Bir maske hangi bitleri saklamak istediğinizi ve hangi bitleri temizlemek istediğinizi tanımlar.
Maskeleme, bir değere maske uygulama eylemidir. Bu, aşağıdakileri yaparak gerçekleştirilir:

*Değerdeki bitlerin bir alt kümesini çıkarmak için Bitwise ANDing
*Değerdeki bitlerin bir alt kümesini ayarlamak için Bitwise ORing
*Değerdeki bitlerin bir alt kümesini değiştirmek için Bitsel XORing

Sayıların ikilik (binary), onluk (decimal) ve on altılık (hexadecimal) 
tabanda ifade edilişleri ve maskeleme yardımıyla veri işleme yöntemleri.

"""
# thresh_img2 = thresh_img2.reshape(thresh_img2.shape[0],thresh_img2.shape[1]*thresh_img2.shape[2])
# thresh_img2 = cv2.resize(thresh_img2, (64,64), interpolation = cv2.INTER_AREA)
print(external_contour.shape)
print(image.shape)
# external_contour = external_contour.reshape(256,256,1)
print(external_contour.shape)
mask = np.zeros((256,256), dtype=np.uint8)
mask = cv2.circle(mask, (256,256), 225, (255,255,255), -1) 
#print(mask.shape)
# Mask input image with binary mask
result = cv2.bitwise_and(thresh_img,thresh_img,image,mask=None)
cv2.imshow('Yaprak',result)
cv2.waitKey(0)                        
# Color background white
#result[mask==0] = 255 # Optional
plt.show()
plt.imshow(result)
plt.title("Final Segmented")
plt.show()
print()
print("************************************")
print("Segmentation has been successfully applied.")
print()

#%% 2. Bölüm

"""
Bu Bölümde blob işlemi yapıyoruz ve labellerimizin isimlerini belirliyoruz.
Oluşacak olan rectangle'lerin renklerini isteginize göre ayarlayabilirsiniz
"""
img_blob = cv2.dnn.blobFromImage(image, 1/255, (416,416), swapRB=True, crop=False)

labels = ["Healthy","Mild Infected","Severe Infected"]
"""
3 classımız var onları labels adlı dizinin içinde tanımlıyoruz...

"""


colors = ["0,0,255","0,255,255","0,0,255"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors,(3,1))



#%% 3. Bölüm


"""

NOT : NON MAXIMUM SUPPRESSION 

NMS, nesne algılamada belirli bir nesnenin yalnızca bir kez tanımlandığından emin olmak için kullanılır. 
9X9 ızgaralı 100X100 bir görüntü düşünün ve algılamak istediğimiz bir araba var. 
Bu araba birden fazla ızgara hücresinde bulunuyorsa, NMS, bu arabanın ait olduğu tüm adaylar arasında en 
uygun hücreyi belirlememizi sağlar.
NMS'nin çalışma şekli:
→ Önce nesnenin mevcut olma olasılığının (son softmax katmanında hesaplanan) <= 0,6 olduğu tüm hücreleri atar
→ Ardından nesne adayları arasında en büyük olasılığa sahip hücreyi bir tahmin olarak alır


"""

"""
Google Colab ile Tesla GPU'suna bağlanarak eğittiğimiz modelimizin configürasyon ve weights dosyalarının
pathlerini belirtiyoruz.
Yolo algoritmasının Version 4-tiny sürümünü kullandım.

"""
model = cv2.dnn.readNetFromDarknet("yolov4-tiny-leaf.cfg","yolov4-tiny-leaf_last.weights")

layers = model.getLayerNames()
output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer)


"""############## NON-MAXIMUM SUPPRESSION - OPERATION 1 ###################"""
 
ids_list = []
boxes_list = []
confidences_list = []

"""############################ END OF OPERATION 1 ########################"""


#%% 4. Bölüm


for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]
        
        """ Güven aralığı > %80 olan tahminlerin gösterilmesi"""
        if confidence > 0.80:
            
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
            
            start_x = int(box_center_x - (box_width/2))
            start_y = int(box_center_y - (box_height/2))
            
            
            """############## NON-MAXIMUM SUPPRESSION - OPERATION 2 ################### """
            
            
            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
            
            
            """ ############################ END OF OPERATION 2 ######################## """
            
#%% 5. Bölüm            
"""############## NON-MAXIMUM SUPPRESSION - OPERATION 3 ################### """

            
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
     
for max_id in max_ids:
    max_class_id = max_id[0]
    box = boxes_list[max_class_id]
    
    start_x = box[0] 
    start_y = box[1] 
    box_width = box[2] 
    box_height = box[3] 
     
    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]
  
    
    """############################ END OF OPERATION 3 ######################## """
            
    end_x = start_x + box_width
    end_y = start_y + box_height
            
    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]
            
            
    label = "{}: {:.2f}%".format(label, confidence*100)
    print("The Leaf Object was Predicted Successfully with Yolov4-Tiny. {}  ".format(label))
     
            
    cv2.rectangle(image, (start_x,start_y),(end_x,end_y),box_color,1)
    cv2.putText(image,label,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)


cv2.imshow("Detection Window", image)

  

print("Image Processing Project")
print("Muhammet ÇINAR")

print(" THE END ...") 

cv2.waitKey()
cv2.destroyAllWindows()



#%% Son




