from flask import Flask, request, send_file, Response, render_template
from flask_restful import Resource, Api
from bs4 import BeautifulSoup
import pandas as pd
import requests
import csv
import certifi
import urllib.parse
import base64
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from pyramid.arima import auto_arima
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.linear_model import LinearRegression
import numpy as np
import math
import calendar

from time import strptime
from io import StringIO
import io


app=Flask(__name__)

@app.route('/traffics', methods=['POST'])
def upload_traffic_file():
    
    # checking if the file is present or not.
    if 'file' not in request.files:
        return "No file found"
 
    file = request.files['file']
    file.save("datalake/"+file.filename)

    return "traffic file successfully saved"

@app.route('/traffics/<string:monthyear>', methods=['GET'])
def get_traffic_monthyear(monthyear):
    #get traffic from monthyear
    df=pd.pandas.read_csv('./datalake/'+monthyear+'.csv', index_col=False, header=0)
    buffer=StringIO()
    df.to_csv(buffer,encoding='utf-8',index=False)

    return Response(buffer.getvalue(), mimetype='text/csv')

@app.route('/matches', methods=['POST'])
def webscrap_post():
    args=request.form['year']
    tables = pd.read_html("http://www.worldfootball.net/teams/manchester-united/"+args+"/3/")
    df=tables[1]
    df.columns=['Round','date','time','place','logo','Opponent','Results','desc']
    df=df.drop(df.columns[[4,7]], axis=1)
    df = df.dropna(how='any',axis=0) 
    df.to_csv("datalake/"+args+"matches.csv", index = False, sep=',', encoding='utf-8')

    return 'match file succesfully saved'  

@app.route('/matches/<years>')
def get_matches_by_year(years):
    yearafter=str(int(years)+1)
    df=pd.pandas.read_csv('./datalake/'+years+'matches.csv', index_col=False, header=0)
    df2=pd.pandas.read_csv('./datalake/'+yearafter+'matches.csv', index_col=False, header=0)
    df3=df.append(df2)
    # print(df3)
    df3['date'] = pd.to_datetime(df3['date'])
    df4=df3[df3['date'].apply(lambda x:x.strftime('%Y'))==years]
    df4.drop(df4.columns[0],axis=1)    
    print(df4)
    buffer=StringIO()
    df4.to_csv(buffer,encoding='utf-8',index=False)
    return Response(buffer.getvalue(), mimetype='text/csv')

@app.route('/matches/<years>/<place>')
def get_matches_by_year_place(years,place):
    yearafter=str(int(years)+1)
    df=pd.pandas.read_csv('./datalake/'+years+'matches.csv', index_col=False, header=0)
    df2=pd.pandas.read_csv('./datalake/'+yearafter+'matches.csv', index_col=False, header=0)
    df3=df.append(df2)
    print(df3)
    df3['date'] = pd.to_datetime(df3['date'])
    df4=df3[df3['date'].apply(lambda x:x.strftime('%Y'))==years]
    if place == "home" :
        print('you are here')
        df5=df4[df4['place'].str.match("H")]
    elif place =="away":
         df5=df4[df4['place'].str.match("A")]
    else: 
        return 'matches home or away not found'

    df5.drop(df5.columns[0],axis=1)    
    print(df5)
    buffer=StringIO()
    df5.to_csv(buffer,encoding='utf-8',index=False)
    return Response(buffer.getvalue(), mimetype='text/csv')

@app.route('/integrations/hometrafficmatches/<monthyear>')
def get_hometrafficmatches_by_monthyear(monthyear):
    #extrach month and year
    mnth=monthyear[:-4]
    years=monthyear[-4:]
    yearafter=str(int(years)+1)
    numofmonth=strptime(mnth, '%b').tm_mon

    #get data matches by month and year "2012matches.csv"
    df=pd.pandas.read_csv('./datalake/'+years+'matches.csv', index_col=False, header=0)
    df2=pd.pandas.read_csv('./datalake/'+yearafter+'matches.csv', index_col=False, header=0)
    df3=df.append(df2)
    
    #filter df based on arg month and year 
    df3['date'] = pd.to_datetime(df3['date'])
    df4=df3[df3['date'].apply(lambda x:x.strftime('%Y-%m'))==years+'-'+str(numofmonth)]
    df5=df4[df4['place'].str.match("H")]
    df5.drop(df5.columns[0],axis=1)    
    dfmntrahome=df5['date']

    
    #get data traffic based on month and year "Dec2012.csv"
    df=pd.pandas.read_csv('./datalake/'+monthyear+'.csv',index_col=False, header=0)
    df['Sdate'] = pd.to_datetime(df['Sdate'])
    
    #filter all traffic on the home match
    mask=df['Sdate'].isin(dfmntrahome)
    
    #response to browser
    buffer=StringIO()
    df[mask].to_csv(buffer,encoding='utf-8',index=False)
    return Response(buffer.getvalue(), mimetype='text/csv')

@app.route('/integrations/traffics/<monthfrom>/<int:yearfrom>/<monthto>/<int:yearto>', methods=['GET'])
def get_traffics(monthfrom,yearfrom,monthto,yearto):
    
    mnthfrom=strptime(monthfrom, '%b').tm_mon
    mnthto=strptime(monthto, '%b').tm_mon
    
    df=pd.DataFrame()
    months=mnthfrom
    for years in range(yearfrom,yearto+1):
        print("inside years")
        print(years)
        for months in range(mnthfrom,mnthto+1): 
            month=calendar.month_abbr[months]
            dftemp=pd.pandas.read_csv('./datalake/'+month+str(years)+'.csv', index_col=False, header=0)
            df=df.append(dftemp)
        month=calendar.month_abbr[months]
        dftemp=pd.pandas.read_csv('./datalake/'+month+str(years)+'.csv', index_col=False, header=0)
        df=df.append(dftemp)

    df.to_csv("datalake/intraffic"+monthfrom+str(yearfrom)+monthto+str(yearto)+".csv", index = False, sep=',', encoding='utf-8')
    # response to browser
    buffer=StringIO()
    df.to_csv(buffer,encoding='utf-8',index=False)
    return Response(buffer.getvalue(), mimetype='text/csv')

@app.route('/arima', methods=['GET'])
def get_arima():
    
    dateparse = lambda dates: datetime.strptime(dates, '%d/%m/%Y')
    #series = read_csv('Traffic Data 2012-2014.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    series = read_csv('NovDec 2012-2014.csv', parse_dates=['Sdate'], index_col='Sdate',date_parser=dateparse)
    #print(series)

    dat = series.loc[(series['Day'] == 'Wednesday') & (series['LaneNumber'] ==1) & (series['Hour'] == 19)]
    #print(dat)
    train = dat.loc[(dat['Date'] != '28/11/2012') & (dat['Date'] != '05/12/2012') & (dat['Date'] != '26/12/2012') & (dat['Date'] != '04/12/2013')]
    #print(train)
    dtrain = train['Volume']
    #print(dtrain)
    test = dat['Volume']
    #print(test)

    X = dtrain.values
    #print(X)
    Y = test.values
    #print(Y)

    #size = int(len(X) * 0.66)
    #train, test = X[0:size], X[size:len(X)]
    history = [X for X in dtrain]
    predictions = list()

    for t in range(len(Y)):
        model = ARIMA(history, order=(1,0,0))
        #model = auto_arima(history, start_p=1, start_q=1,
                            #max_p=3, max_q=3, m=12,
                            #start_P=0, seasonal=True,
                            #d=1, D=1, trace=True,
                            #error_action='ignore',  
                            #suppress_warnings=True, 
                            #stepwise=True)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = Y[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.1f' % error)
    # plot
    img=io.BytesIO()
    pyplot.plot(Y, color='grey')
    pyplot.plot(predictions, color='blue')
    pyplot.ylabel('Volume')
    pyplot.xlabel('Row Number')
    pyplot.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()
    return '<img src="data:image/png;base64,{}">'.format(plot_url)

#get traffic from monthyear
##@app.route('/arima/traffic/<day>/<int:lanedirection>/<int:hour>/<int:hour2>', methods=['GET'])
##def get_arima_traffics(day,lanedirection,hour, hour2):
@app.route('/arima/traffic/<day>/<int:lanenumber>/<int:hour>/<int:hour2>', methods=['GET'])
def get_arima_traffics(day,lanenumber,hour, hour2):
##@app.route('/arima/traffic/<day>/<int:lanenumber>/<int:hour>', methods=['GET'])
##def get_arima_traffics(day,lanenumber,hour):
##@app.route('/arima/traffic/<day>/<int:hour>/<int:hour2>', methods=['GET'])
##def get_arima_traffics(day, hour, hour2):  
##@app.route('/arima/traffic/<day>/<int:lanedirection>/<int:hour>', methods=['GET'])
##def get_arima_traffics(day, lanedirection, hour): 
##@app.route('/arima/traffic/<day>/<int:hour>', methods=['GET'])
##def get_arima_traffics(day, hour): 
    # df=pd.pandas.read_csv('./datalake/'+monthyear+'.csv', index_col=False, header=0)
    # buffer=StringIO()
    # df.to_csv(buffer,encoding='utf-8',index=False)
    # return Response(buffer.getvalue(), mimetype='text/csv')

    dateparse = lambda dates: datetime.strptime(dates, '%d/%m/%Y %H:%M')
    series = read_csv('./datalake/Traffic Data 2012-2014.csv', parse_dates=['Sdate'], index_col='Sdate',date_parser=dateparse)
   
    ##dat = series.loc[(series['Day'] == day) & (series['LaneDirection'] == lanedirection) & (series['Hour'] >= hour) & (series['Hour'] <= hour2)]
    dat = series.loc[(series['Day'] == day) & (series['LaneNumber'] == lanenumber) & (series['Hour'] >= hour) & (series['Hour'] <= hour2)]
    ##dat = series.loc[(series['Day'] == day) & (series['LaneNumber'] == lanenumber) & (series['Hour'] == hour)]
    ##dat = series.loc[(series['Day'] == day) & (series['Hour'] >= hour) & (series['Hour'] <= hour2)]
    ##dat = series.loc[(series['Day'] == day) & (series['LaneDirection'] == lanedirection) & (series['Hour'] == hour)]
    ##dat = series.loc[(series['Day'] == day) & (series['Hour'] == hour)]
    
    ##dat['Outlier']=np.where(dat['Date']=='28/11/2012', 1, 0)
    ##train = dat.loc[(dat['Date'] != '28/11/2012') & (dat['Date'] != '05/12/2012') & (dat['Date'] != '26/12/2012') & (dat['Date'] != '04/12/2013')]
    ##test = dat.loc[(dat['Date'] != '05/12/2012') & (dat['Date'] != '26/12/2012') & (dat['Date'] != '04/12/2013')]
    dat['Outlier']=np.where(dat['Date']=='10/12/2013', 1, 0)
    train = dat.loc[(dat['Date'] != '10/12/2013') & (dat['Date'] != '02/12/2014')]
    test = dat.loc[(dat['Date'] != '02/12/2014')]
    #test=dat

    dtrain = train['Volume']
    dtest = test['Volume']

    X = dtrain.values
    Y = dtest.values
    
    history = [X for X in dtrain]
    predictions = list()

    for t in range(len(Y)):
        model = auto_arima(history, start_p=1, start_q=1,
                            max_p=3, max_q=3, m=1,
                            start_P=0, seasonal=False,
                            d=1, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
        print(model.summary())
        output = model.predict()
        pred = output[0]
        predictions.append(pred)
        hist = Y[t]
        history.append(hist)
        print('predicted=%f, expected=%f' % (pred, hist))

    df=test[['Day', 'LaneNumber', 'DirectionDescription', 'Volume', 'AvgSpeed', 'Outlier']]
    df = df.rename_axis(None)
    df.columns.name='Sdate'
    df.loc[ : ,'VolumePrediction']=predictions
    df.insert(0, 'Row', range(0, 0+len(df)))

    x=df['Volume']
    y=df['VolumePrediction']
    #a=df['Volume'].values
    #b=df['VolumePrediction'].values[:,np.newaxis]
    #model = LinearRegression()
    #model.fit(b, a)

    #pyplot.scatter(b,a,color='r')
    #pyplot.plot(b, model.predict(b),color='k')
    #pyplot.show()

    for i in range(len(df)):
        dist=x-y
        dist2=dist**2

    df['Distance']=dist
    df['SquareDistance']=dist2
    sse= df['SquareDistance'].sum()
    stdev=math.sqrt(sse/(len(df)-2))
    threshold=2*stdev

    def f(row):
        if row['Distance'] > threshold:
            val = 1
        else:
            val = 0
        return val

    df['OutlierPrediction'] = df.apply(f, axis=1)
    df2=df.loc[df['OutlierPrediction'] == 1]
    print(df)

    #BalancedAccuracy = balanced_accuracy_score(df['Outlier'], df['OutlierPrediction'])
    tn, fp, fn, tp = confusion_matrix(df['Outlier'].values, df['OutlierPrediction'].values).ravel()
    sensitivity=tp/(tp+fn)
    #sensitivity=recall_score(df['Outlier'], df['OutlierPrediction'], average='weighted')
    specificity=tn/(fp+tn)
    BalancedAccuracy=(sensitivity+specificity)/2
    FPRate = fp/(fp+tn)
    print('Balanced Accuracy=%.2f, FP Rate=%.2f' % (BalancedAccuracy, FPRate))

    #plot
    img=io.BytesIO()
    markers_on = [df.loc[df['OutlierPrediction'] == 1, 'Row']]
    pyplot.plot(Y, color='blue', marker='D', markevery=markers_on)
    #pyplot.plot(Y, color='blue')
    pyplot.plot(predictions, color='red')
    pyplot.ylabel('Volume')
    pyplot.xlabel('Row Number')
    #pyplot.show()
    pyplot.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()
    rsp='<img src="data:image/png;base64,{}">'.format(plot_url)
    
    # return rsp
    return render_template("home.html", graph=rsp, data=df2.to_html())

@app.route('/autoarima/traffic/<int:day>/<int:month>/<int:year>/<int:lanenumber>/<int:hour>/<int:hour2>', methods=['GET'])
def get_autoarima_traffics(date,lanenumber,hour, hour2):
    
    dateparse = lambda dates: datetime.strptime(dates, '%d/%m/%Y %H:%M')
    series = read_csv('./datalake/Traffic Data 2012-2014.csv', parse_dates=['Sdate'], index_col='Sdate',date_parser=dateparse)
    wday = date(day,month,year).weekday()
    date_matches_list = series.loc[series['Outlier']==1 & series[Iday]==wday, 'Date'].iloc[0]
    date_matches_train = date_matches_list[date_matches_list['Date'] != day/month/year].iloc[0]
    
    test = series.loc[series['Day'] == wday) & (series['Date'] == day/month/year) & (series['LaneNumber'] ==lanenumber) & (series['Hour'] >= hour) & (series['Hour'] <= hour2)]
    train = series.loc[series['Day'] == day) & (series['Date'] != date_matches_train) & (series['LaneNumber'] ==lanenumber) & (series['Hour'] >= hour) & (series['Hour'] <= hour2)]

    dtrain = train['Volume']
    dtest = test['Volume']

    X = dtrain.values
    Y = dtest.values
    
    history = [X for X in dtrain]
    predictions = list()

    for t in range(len(Y)):
        model = auto_arima(history, start_p=1, start_q=1,
                            max_p=3, max_q=3, m=1,
                            start_P=0, seasonal=False,
                            d=1, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
        print(model.summary())
        output = model.predict()
        pred = output[0]
        predictions.append(pred)
        hist = Y[t]
        history.append(hist)
        print('predicted=%f, expected=%f' % (pred, hist))

    df=test[['Day', 'LaneNumber', 'DirectionDescription', 'Volume', 'AvgSpeed', 'Outlier']]
    df = df.rename_axis(None)
    df.columns.name='Sdate'
    df.loc[ : ,'VolumePrediction']=predictions
    df.insert(0, 'Row', range(0, 0+len(df)))

    x=df['Volume']
    y=df['VolumePrediction']

    for i in range(len(df)):
        dist=x-y
        dist2=dist**2

    df['Distance']=dist
    df['SquareDistance']=dist2
    sse= df['SquareDistance'].sum()
    stdev=math.sqrt(sse/(len(df)-2))
    threshold=2*stdev

    def f(row):
        if row['Distance'] > threshold:
            val = 1
        else:
            val = 0
        return val

    df['OutlierPrediction'] = df.apply(f, axis=1)
    df2=df.loc[df['OutlierPrediction'] == 1]
    print(df)

    #BalancedAccuracy = balanced_accuracy_score(df['Outlier'], df['OutlierPrediction'])
    tn, fp, fn, tp = confusion_matrix(df['Outlier'].values, df['OutlierPrediction'].values).ravel()
    sensitivity=tp/(tp+fn)
    #sensitivity=recall_score(df['Outlier'], df['OutlierPrediction'], average='weighted')
    specificity=tn/(fp+tn)
    BalancedAccuracy=(sensitivity+specificity)/2
    FPRate = fp/(fp+tn)
    print('Balanced Accuracy=%.2f, FP Rate=%.2f' % (BalancedAccuracy, FPRate))

    #plot
    img=io.BytesIO()
    markers_on = [df.loc[df['OutlierPrediction'] == 1, 'Row']]
    pyplot.plot(Y, color='blue', marker='D', markevery=markers_on)
    pyplot.plot(predictions, color='red')
    pyplot.ylabel('Volume')
    pyplot.xlabel('Row Number')
    #pyplot.show()
    pyplot.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()
    rsp='<img src="data:image/png;base64,{}">'.format(plot_url)
    
    # return rsp
    return render_template("home.html", graph=rsp, data=df2.to_html())

@app.after_request
def add_header(r):
  
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

app.run(port=5000)