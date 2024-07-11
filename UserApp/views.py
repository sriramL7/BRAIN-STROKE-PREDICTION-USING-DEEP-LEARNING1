from django.shortcuts import render
import sqlite3
import joblib
from django.core.files.storage import FileSystemStorage
import cv2
import imutils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
import os
import numpy as np
# Create your views here.
def index(request):
    return render(request, "UserApp/index.html")

def login(request):
    return render(request, "UserApp/Login.html")

def loglction(request):
    username=request.POST.get('username')
    password=request.POST.get('password')
    con = sqlite3.connect("brainstroke.db")
    cur=con.cursor()
    cur.execute("select *  from user where username='"+username+"'and password='"+password+"'")
    data=cur.fetchone()
    if data is not None:
        request.session['user']=username
        request.session['userid']=data[0]
        return render(request,'UserApp/UserHome.html')
    else:
        context={'data':'Login Failed ....!!'}
        return render(request,'UserApp/Login.html',context)
def userhome(request):
    return render(request,'UserApp/UserHome.html')

def register(request):
    return render(request, "UserApp/Register.html")
def regaction(request):
    name=request.POST['name']
    email=request.POST['email']
    mobile=request.POST['mobile']
    address=request.POST['address']
    username=request.POST['username']
    password=request.POST['password']

    con = sqlite3.connect("brainstroke.db")
    cur=con.cursor()
    #cur.execute("CREATE TABLE user (ID INTEGER PRIMARY KEY AUTOINCREMENT,name varchar(100),email varchar(100),mobile varchar(100),address varchar(100) ,username varchar(100),password varchar(100))")
    i=cur.execute("insert into user values(null,'"+name+"','"+email+"','"+mobile+"','"+address+"','"+username+"','"+password+"')")
    con.commit()
    con.close()
    if i == 0:
        context={'data':'Registration Failed...!!'}
        return render(request,'UserApp/Register.html',context)
    else:
        context={'data':'Registration Successful...!!'}
        return render(request,'UserApp/Register.html',context)


def viewprofile(request):
    uid=str(request.session['userid'])
    con = sqlite3.connect("brainstroke.db")
    cur=con.cursor()
    cur.execute("select * from user where id='"+uid+"'")
    data=cur.fetchall()
    strdata="<table border=1><tr><th>Name</th><th>Email</th><th>Mobile</th><th>Address</th><th>Username</th></tr>"
    for i in data:
        strdata+="<tr><td>"+str(i[1])+"</td><td>"+str(i[2])+"</td><td>"+str(i[3])+"</td><td>"+str(i[4])+"</td><td>"+str(i[5])+"</td></tr>"
    context={'data':strdata}
    return render(request,'UserApp/ViewProfile.html',context)

def uploadImage(request):
    return render(request,'UserApp/Upload.html')


global filename, uploaded_file_url


def imageAction(request):
    global filename, uploaded_file_url
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        location = myfile.name
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print("BASE_DIR: "+BASE_DIR)
        print("uploaded_file: "+uploaded_file_url)

        imagedisplay = cv2.imread(BASE_DIR + "/" + uploaded_file_url)
        cv2.imshow('uploaded Image', imagedisplay)
        cv2.waitKey(0)
    context = {'data': 'Test Image Uploaded Successfully'}
    return render(request, 'UserApp/Upload.html', context)


def brainstrokepredict(request):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imagetest = image.load_img(BASE_DIR + "/" + uploaded_file_url, target_size=(48, 48))
    imagetest = image.img_to_array(imagetest)
    imagetest = np.expand_dims(imagetest, axis=0)
    classifier = Sequential()
    classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 3)))
    classifier.add(MaxPooling2D((2, 2)))
    classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
    classifier.add(MaxPooling2D((2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(activation="relu", units=128))
    classifier.add(Dense(activation="softmax", units=2))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.load_weights('model/brain_model.h5')
    pred = classifier.predict(imagetest)
    print(str(pred) + " " + str(np.argmax(pred)))
    predict = np.argmax(pred)

    if predict == 1:
        data = "Brain Stroke Predicted"
    else:
        data = "Brain Stroke Not Predicted"


    imagedisplay = cv2.imread(BASE_DIR + "/" + uploaded_file_url)
    oring = imagedisplay.copy()
    output = imutils.resize(oring, width=400)

    cv2.putText(output, data, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Predicted image result", output)
    cv2.waitKey(0)
    return render(request,'UserApp/UserHome.html')




