import sys
import cv2

speed = 0.2
axonNum = 10 # axonNum = кол-во символов для расознования (learn{n} должно быть столько же)
sensorX = 10
senxorY = 10 #Всего сенсоров = sensorX*senxorY

class Perceptron():
    def __init__(self):
        self.weights = [0.5]*sensorX*senxorY

class Web():
    def __init__(self):
        self.perceptrons = [Perceptron() for i in range(axonNum)]
        self.results = [0]*axonNum
        
    def learn(self, inputt, number, speed): #number = [0..9], speed = [0..1]
        for i in range(sensorX):
            for j in range(senxorY):
                self.perceptrons[number].weights[i*10+j] += speed * (inputt[i][j] - 127.5)/255
        minn = min(self.perceptrons[number].weights)
        maxx = max(self.perceptrons[number].weights)
        for i in range(sensorX):
            for j in range(senxorY):
                self.perceptrons[number].weights[i*10+j]=(self.perceptrons[number].weights[i*10+j]-minn)/(maxx - minn)

    def result(self, inputt):
        for number in range(axonNum):
            for i in range(sensorX):
                for j in range(senxorY):
                    self.results[number] += self.perceptrons[number].weights[i*10+j]*(inputt[i][j])

A = Web()

img = [cv2.imread('learn{}.png'.format(i)) for i in range(axonNum)]

for num, im in zip(range(axonNum), img):
    im3 = im.copy()

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt)>2000:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>28:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(sensorX,senxorY))
                #cv2.imshow('norm',im)
                A.learn(roismall, num, speed)
                print("training {} complete".format(num))
                
print("training complete")
print("--------------------------------------------------------------------------------")
for i in range(axonNum):
    print(A.perceptrons[i].weights)
print("--------------------------------------------------------------------------------")

im = cv2.imread('example5.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)>1000:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(sensorX,senxorY))
            cv2.imshow('norm',im)
            print(roismall)
            A.result(roismall)
            print("training complete")

print(A.results)
print(A.results.index(max(A.results)))
