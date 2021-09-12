from flask import Flask,render_template,request,flash,redirect,url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from facial_emotion_recognition import EmotionRecognition
from deepface import DeepFace
import cv2
import os


UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def front_page():
    return render_template('frontpage.html')
@app.route('/dementia',methods=['GET','POST'])
def dementia_page():
    if request.method == 'GET':
        return render_template('dementia.html')
    else:
        age = request.form['age']
        sex = request.form['sex']
        educ = request.form['educ']
        ses = request.form['ses']
        mmse = request.form['mmse']
        etiv = request.form['etiv']
        nwbv = request.form['nwbv']
        asf = request.form['asf']
        df = pd.read_csv('oasis_longitudinal.csv')
        df["SES"].fillna(df["SES"].median(), inplace=True)
        df["MMSE"].fillna(df["MMSE"].mean(), inplace=True)
        df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
        group_map = {"Demented": 1, "Nondemented": 0}
        df['Group'] = df['Group'].map(group_map)
        df['M/F'] = df['M/F'].replace(['F','M'], [0,1])
        for column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
        feature_col_names = ["M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
        predicted_class_names = ['Group']

        X = df[feature_col_names].values
        y = df[predicted_class_names].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        rfc = RandomForestClassifier(n_estimators = 200)
        rfc.fit(X_train,y_train.ravel())
        testing_data = np.array([[sex,age,educ,ses,mmse,etiv,nwbv,asf]])
        prediction = rfc.predict(testing_data)
        if prediction[0] == 1:
            senddata="Dementia disease is there"
        else:
            senddata="dementia disease is not there"

        return render_template('result.html',resultvalue=senddata)

@app.route('/image',methods=['GET','POST'])
def image_page():
    if request.method == 'GET':
        return render_template('image.html')
    else:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('image_page'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(url_for('image_page'))

        if file and allowed_file(file.filename):
            
            filename = secure_filename(file.filename)
            fullname=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fullname)
            try:
                image = cv2.imread(fullname)
                result = DeepFace.analyze(image,actions=['emotion'])
                print(result)
                return render_template('resultimage.html',filename=filename,resultemotion=result['dominant_emotion'])
            except:
                flash('Please Upload a image which is Containing Face Clearly.')
                return redirect(url_for('image_page'))
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(url_for('image_page'))

       
if __name__ == '__main__':
    app.run()