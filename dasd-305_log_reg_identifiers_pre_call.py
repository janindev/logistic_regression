import os.path
import matplotlib.pyplot as pplt
import numpy as np
import pandas as pd
import openpyxl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import sys


class LogReg:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Obtain Classes
    def clss(self):
        model = LogisticRegression(solver='sag').fit(x, y)
        clss = model.classes_  
        print('values of y:', clss, '\n')

    # Obtain b0
    def intc(self):
        model = LogisticRegression(solver='sag').fit(x, y)
        intc = model.intercept_  
        print('intercept:', intc, '\n')

    # Obtain Coefficients
    def coef(self):
        model = LogisticRegression(solver='sag').fit(x, y)
        coef = model.coef_  
        print('coefficients:', coef, '\n')

    # Obtain Probability [0 or 1] Predictions
    def predp(self):
        model = LogisticRegression(solver='sag').fit(x, y)
        predp = model.predict_proba(x)  
        print('predicted response (probability):', predp, '\n')        

    # Obtain Actual Predictions
    def pred(self):
        model = LogisticRegression(solver='sag').fit(x, y)
        pred = model.predict(x)  
        print('predicted response (actual):', pred, '\n')

    # Confusion Matrix
    def conf(self):
        model = LogisticRegression(solver='sag').fit(x, y)
        pred = model.predict(x)  
        conf = confusion_matrix(y, pred)
        print('confusion matrix:', '\n', conf, '\n')
        

    # Obtain Model Score
    def scr(self):
        model = LogisticRegression(solver='sag').fit(x, y)
        scr = model.score(x,y)
        print('score:', scr, '\n')        


# Get actuals into dataframe
location = 'C:\\Users\\970jwillems\\OneDrive - Sonova\\Development - Data Analysis\\Statistical Models\\Logistic Regression\\'
file =  'dasd-305_identifiers_actuals_pre_call_dropout.xlsx'
filepath = os.path.join(location, file)
df = pd.read_excel(filepath)

# Define and shape model data
x0 = df['weekday_flag'].to_numpy()
x1 = df['late_hour_flag'].to_numpy()

x2 = df['channel_facebook_flag'].to_numpy()
x3 = df['channel_taboola_flag'].to_numpy()
x4 = df['channel_outbrain_flag'].to_numpy()
x5 = df['native_display_flag'].to_numpy()

x6 = df['form_st_flag'].to_numpy()
x7 = df['form_sa_flag'].to_numpy()
x8 = df['form_sl_flag'].to_numpy()

x9 = df['bucket_1ef_flag'].to_numpy()
x10 = df['bucket_1abcd_flag'].to_numpy()
x16 = df['bucket_3_flag'].to_numpy()

x11 = df['has_hea_flag'].to_numpy()
x12 = df['has_ent_prescription_flag'].to_numpy()
x17 = df['is_old_enough_flag'].to_numpy()
x13 = df['post_qualification_completed_flag'].to_numpy()
x14 = df['post_qualification_started_flag'].to_numpy()

x15 = df['mobile_device_flag'].to_numpy()

x = np.vstack((x0
               , x1
               , x2
               , x3
               , x4
               #, x5
               , x6
               , x7
               , x8
               #, x9
               , x10
               , x16
               , x11
               #, x12
               , x17 #very minimal improvement
               , x13
               #, x14
               , x15
               )).T

y = df['drop_out_flag_pre_call'].to_numpy()


# Model outcome/attributes
lgm = LogReg(x,y)
lgm.intc()
lgm.coef()
#lgm.predp()
#lgm.pred()
lgm.conf()
lgm.scr()
