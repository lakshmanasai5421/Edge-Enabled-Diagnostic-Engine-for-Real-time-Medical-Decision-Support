from imblearn.over_sampling import SMOTE
from flask import Flask, render_template, request
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import pickle
import warnings
warnings.filterwarnings('ignore')

# custom module for encryption
from LPME import privacyPreservingTrain, privacyPreservingTest

app = Flask(__name__)
app.secret_key = 'dropboxapp1234'
global classifier

# ----------------- ROUTES -----------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/Signup')
def Signup():
    return render_template("Signup.html")

@app.route('/Login')
def Login():
    return render_template("Login.html")

# ---------- User Authentication ----------
@app.route('/SignupAction', methods=['POST'])
def SignupAction():
    username = request.form['t1']
    password = request.form['t2']
    contact = request.form['t3']
    email = request.form['t4']
    address = request.form['t5']
    users = []

    if os.path.exists('static/user.db'):
        with open('static/user.db', 'rb') as file:
            users = pickle.load(file)

    for u in users:
        if u.split(",")[0] == username:
            return render_template("Signup.html", error=f"{username} username already exists")

    users.append(f"{username},{password},{contact},{email},{address}")
    with open('static/user.db', 'wb') as file:
        pickle.dump(users, file)
    return render_template("Signup.html", error='Signup successful')

@app.route('/UserLogin', methods=['POST'])
def UserLogin():
    username = request.form['t1']
    password = request.form['t2']
    if os.path.exists('static/user.db'):
        with open('static/user.db', 'rb') as file:
            users = pickle.load(file)
        for u in users:
            name, pw, *_ = u.split(",")
            if name == username and pw == password:
                return render_template("AdminScreen.html", error=f"Welcome {username}")
    return render_template("Login.html", error='Invalid Login')

# ---------- Dataset Selection ----------
@app.route('/selectdata')
def selection():
    return render_template('AdminScreen.html', select=True)

# ---------- Data Encryption ----------
@app.route("/Encrypt", methods=["POST"])
def Encrypt():
    dataset_id = request.form.get('id')
    if dataset_id == 'heart':
        src, dst = 'Dataset/heart.csv', 'EncryptedData/enc_heart.csv'
    elif dataset_id == 'hypothyroid':
        src, dst = 'Dataset/hypothyroid.csv', 'EncryptedData/enc_hypothyroid.csv'
    else:
        return render_template("AdminScreen.html", error="Invalid dataset ID provided or missing.")

    privacyPreservingTrain(src, dst)
    plain = pd.read_csv(src)
    encrypted = pd.read_csv(dst, encoding='latin1')

    output = '<table border="1" align="center">'
    output += '<tr><th>Plain Dataset</th><th>Encrypted Dataset</th></tr>'
    output += f'<tr><td>{plain.head().to_html()}</td><td>{encrypted.head().to_html()}</td></tr>'
    output += '</table>'
    return render_template("AdminScreen.html", error=output)

# ---------- Model Training ----------
@app.route("/TrainML")
def TrainML():
    global classifier

    # Load dataset
    dataset = pd.read_csv('EncryptedData/enc_heart.csv', encoding='latin1')
    dataset.fillna(0, inplace=True)
    dataset = dataset.values
    cols = dataset.shape[1] - 1
    X = normalize(dataset[:, 0:cols])
    Y = dataset[:, cols]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    output = '<table border="1" align="center">'
    output += '<tr><th>Algorithm Name</th><th>Accuracy</th><th>Recall</th><th>Specificity</th></tr>'

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred) * 100
    nb_recall = recall_score(y_test, nb_pred, average='macro') * 100
    cm = confusion_matrix(y_test, nb_pred)
    nb_spec = cm[1, 1] / (cm[1, 0] + cm[1, 1]) * 100 if (cm[1, 0] + cm[1, 1]) != 0 else 0
    joblib.dump(nb, 'Model/NaiveBayes_weights.pkl')
    output += f'<tr><td>NaiveBayes</td><td>{nb_acc:.2f}</td><td>{nb_recall:.2f}</td><td>{nb_spec:.2f}</td></tr>'

    # XGBoost
    classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    classifier.fit(X_train, y_train)
    xgb_pred = classifier.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred) * 100
    xgb_recall = recall_score(y_test, xgb_pred, average='macro') * 100
    cm = confusion_matrix(y_test, xgb_pred)
    xgb_spec = cm[1, 1] / (cm[1, 0] + cm[1, 1]) * 100 if (cm[1, 0] + cm[1, 1]) != 0 else 0
    joblib.dump(classifier, 'Model/XGBoost_weights.pkl')
    output += f'<tr><td>XGBoost</td><td>{xgb_acc:.2f}</td><td>{xgb_recall:.2f}</td><td>{xgb_spec:.2f}</td></tr>'
    output += '</table>'

    # Save feature columns
    feature_cols = pd.read_csv('EncryptedData/enc_heart.csv', encoding='latin1').dropna(axis=1).columns[:-1].tolist()
    joblib.dump(feature_cols, 'Model/heart_feature_cols.pkl')

    return render_template("AdminScreen.html", error=output)

# ---------- Prediction ----------
@app.route('/selecttest')
def testdata():
    return render_template("AdminScreen.html", Predictselection=True)

@app.route('/Predict', methods=['POST'])
def Predict():
    dataset_id = request.form.get('id')
    testfile = request.files['file']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        testfile.save(temp_file.name)
        testdata = pd.read_csv(temp_file.name)
    os.unlink(temp_file.name)

    # Load correct model and feature columns
    if dataset_id == 'heart':
        model_path = 'Model/NaiveBayes_weights.pkl'
        feature_cols_path = 'Model/heart_feature_cols.pkl'
        labels = ['No Heart Disease', 'Heart Disease']
    elif dataset_id == 'hypothyroid':
        model_path = 'Model/XGBoostClassifier_weights.pkl'
        feature_cols_path = 'Model/XGBoostClassifier_features.pkl'
        labels = ['Thyroid not detected', 'Thyroid detected']
    else:
        return render_template("AdminScreen.html", error="Invalid dataset selected.")

    if not os.path.exists(model_path) or not os.path.exists(feature_cols_path):
        return render_template("AdminScreen.html", error="Model or feature list not found. Please train first.")

    feature_cols = joblib.load(feature_cols_path)
    model = joblib.load(model_path)

    # Align test data columns
    for col in feature_cols:
        if col not in testdata.columns:
            testdata[col] = 0  # add missing columns
    testdata = testdata[feature_cols]  # select only trained features

    # Encode categorical columns
    for col in testdata.select_dtypes(include='object').columns:
        le = LabelEncoder()
        testdata[col] = le.fit_transform(testdata[col])

    # Normalize numeric data
    test = normalize(testdata)

    preds = model.predict(test)
    data = []
    for i, p in enumerate(preds):
        msg = f"Row {i}: ************************************************** {labels[int(p)]}"
        data.append({'row': testdata.iloc[i], 'message': msg})

    return render_template("AdminScreen.html", data=data)

# ---------- Thyroid Dataset Auto-training ----------
df = pd.read_csv('Dataset/hypothyroid.csv', encoding='latin1')
df.drop_duplicates(inplace=True)
le = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])
x = df.drop('binaryClass', axis=1)
y = df['binaryClass']

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def calculateMetrics(name, pred, true):
    p = precision_score(true, pred, average='macro') * 100
    r = recall_score(true, pred, average='macro') * 100
    f = f1_score(true, pred, average='macro') * 100
    a = accuracy_score(true, pred) * 100
    print(f"\n{name} => Accuracy: {a:.2f} | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f:.2f}")
    print(classification_report(true, pred))

# Ridge Classifier
ridge_path = 'Model/RidgeClassifier_weights.pkl'
if os.path.exists(ridge_path):
    ridge = joblib.load(ridge_path)
else:
    ridge = RidgeClassifier()
    ridge.fit(X_train, y_train)
    joblib.dump(ridge, ridge_path)
calculateMetrics("RidgeClassifier", ridge.predict(X_test), y_test)

# XGBoost Classifier
xgb_path = 'Model/XGBoostClassifier_weights.pkl'
xgb_features_path = 'Model/XGBoostClassifier_features.pkl'
joblib.dump(x.columns.tolist(), xgb_features_path)  # save feature columns
if os.path.exists(xgb_path):
    xgb_model = joblib.load(xgb_path)
else:
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, xgb_path)
calculateMetrics("XGBoostClassifier", xgb_model.predict(X_test), y_test)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
