import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
import numpy as np
import xlrd
import csv
#from flask import Flask, render_template, request
from flask_dropzone import Dropzone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
import itertools
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn import linear_model
from sklearn.linear_model import ARDRegression, LinearRegression
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.ensemble import VotingRegressor




basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_MAX_FILE_SIZE=1024,  # set max size limit to a large number, here is 1024 MB
    DROPZONE_TIMEOUT=5 * 60 * 1000  # set upload timeout to a large number, here is 5 minutes
)

dropzone = Dropzone(app)
model = pickle.load(open('model.pkl', 'rb'))

filename = []
@app.route('/l', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        filename.append(f.filename)
        f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
        print (f.filename)
    return render_template('index2.html')
@app.route('/predict2', methods=['POST', 'GET'])
def predict2():
    #print ('/loader/'+filename[0])
    #file = request.files['file']
    #print (os.path.join(app.config['UPLOADED_PATH'], filename[0]))
    tip = pd.read_excel((os.path.join(app.config['UPLOADED_PATH'],'х СТЕ_с_характеристиками.xlsx')))
    #print (tip)
    tip = tip[['Id', 'Вид продукции']]
    KC = pd.read_excel((os.path.join(app.config['UPLOADED_PATH'],filename[0])))
    #print (KC)
    KC = pd.concat([KC, tip.reindex(KC.index)], axis=1)
    KC = KC[KC['Конечная цена'] != 'NULL']
    KC['TRUE_NAME'] = (KC['Наименование товара'].str.rsplit(n=1, expand=True))[0]
    KC = KC.fillna("NULL")
    KC['Идентификатор СТЕ'].replace('NULL', 0).astype(int)
    try:
        X = KC[['Закон основание', 'Начальная стоимость', 'Итоговоя стоимость', 'Статус', 'Количество снижений', 'Процент снижения', 'Количество участников КС', 'Ставка победителя', 'Статус контракта', 'Количество товара', 'Начальная цена за единицу', 'Вид', 'Вид продукции']]# 'Наименование товара','Идентификатор СТЕ','TRUE_NAME',
    except KeyError:
        X = KC[['Статус', 'Количество снижений', 'Процент снижения', 'Количество участников КС', 'Ставка победителя', 'Статус контракта', 'Количество товара', 'Начальная цена за единицу', 'Вид', 'Вид продукции']]# 'Наименование товара','Идентификатор СТЕ','TRUE_NAME',
    
    
    y = KC['Ставка победителя']#'Конечная цена']
    X['Процент снижения'] = X['Процент снижения'].replace('NULL', 0)
    X['Ставка победителя'] = X['Ставка победителя'].replace('NULL', 0)
    y = y.replace('NULL', 0)
    categorical_feature_mask = X.dtypes==object
    categorical_cols = X.columns[categorical_feature_mask].tolist()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    from sklearn.preprocessing import OneHotEncoder
    X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))
    ohe = OneHotEncoder (categorical_features = categorical_feature_mask, sparse = False )
    X_ohe = ohe.fit_transform(X)
    X_dict = X.to_dict(orient = 'records')
    dv_X = DictVectorizer(sparse = False)
    X_encoded = dv_X.fit_transform(X_dict)
    vocab = dv_X.vocabulary_
    X = pd.get_dummies(X, prefix_sep = '_', drop_first = True)
    print (X)
    
    
    return render_template('index.html', name_file = filename[0])










@app.route('/')
def home():
    return render_template('index.html')

@app.route('/file_predict',methods=['POST'])
def file_predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Стоимость равна {}'.format(output))

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Стоимость равна {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)