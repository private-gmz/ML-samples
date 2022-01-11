import emotion_info
import cv2
import time
import imutils
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup as SOUP
import re
import requests as HTTP

from utils import draw_rectangles, read_image, prepare_image
import os
from flask import Flask,jsonify,request,render_template
app = Flask(__name__)

# initialize flask application
app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')
 
# refer to: https://www.geeksforgeeks.org/movie-recommendation-based-emotion-python/
# Sad – Drama
# Disgust – Musical
# Anger – Family
# Anticipation – Thriller
# Fear – Sport
# Enjoyment – Thriller
# Trust – Western
# Surprise – Film-Noir 
def get_movies(emotion):
    urlhere = 'http://www.imdb.com/search/title/?title_type=feature&sort=moviemeter, asc'
    if(emotion == "sad"):
        urlhere = 'http://www.imdb.com/search/title?genres=drama&title_type=feature&sort=moviemeter, asc'
    elif(emotion == "disgust"):
        urlhere = 'http://www.imdb.com/search/title?genres=musical&title_type=feature&sort=moviemeter, asc'
    elif(emotion == "angry"):
        urlhere = 'http://www.imdb.com/search/title?genres=family&title_type=feature&sort=moviemeter, asc'
    elif(emotion == "fear"):
        urlhere = 'http://www.imdb.com/search/title?genres=sport&title_type=feature&sort=moviemeter, asc'
    elif(emotion == "happy"):
        urlhere = 'http://www.imdb.com/search/title?genres=comedy&title_type=feature&sort=moviemeter, asc'
    elif(emotion == "neutral"):
        urlhere = 'http://www.imdb.com/search/title/?title_type=feature&sort=moviemeter, asc'
    elif(emotion == "surprise"):
        urlhere = 'http://www.imdb.com/search/title?genres=film_noir&title_type=feature&sort=moviemeter, asc'


    response = HTTP.get(urlhere)
    data = response.text

    soup = SOUP(data, "lxml")

    title = soup.find_all("a", attrs = {"href" : re.compile(r'\/title\/tt+\d*\/')})
    return title
     
@app.route('/webcam', methods=['POST'])
def webcam():
    # ----------------------------- webcam -----------------------------
    cam = cv2.VideoCapture(0)
    while True:
        star_time = time.time()
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=720)

        out,num_faces = emotion_info.get_face_info(frame)
        res_img = emotion_info.bounding_box(out,frame)
        end_time = time.time() - star_time
        FPS = 1/end_time
        cv2.putText(res_img,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('Face info',res_img)
        if cv2.waitKey(1) &0xFF == ord('q'):
            img = prepare_image(res_img)
            cam.release()
            cv2.destroyAllWindows()
            cus_emotion = out[0]['emotion']
            titles = get_movies(cus_emotion)
            count = 0
            recommendation = []
            for i in titles:
                tmp = str(i).split('>')
                if(len(tmp) == 3):
                   recommendation.append(tmp[1][:-3])
                if(count > 25):
                   break
                count+=1
            return render_template('index.html',emotion = cus_emotion, recommendations = recommendation, image = img)
            break
    

            
if __name__ == '__main__':
    app.run(debug=True,
            use_reloader=True,
            port=4000)