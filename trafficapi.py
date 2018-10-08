from flask import Flask, request, send_file, Response, render_template
from flask_restful import Resource, Api
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.recurrent import LstmAutoEncoder
import pandas as pd
import requests
import csv
import certifi
import urllib.parse
import base64
import time
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
#from statsmodels.tsa.arima_model import ARIMA
from pyramid.arima import auto_arima
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.metrics import confusion_matrix
#from sklearn.linear_model import LinearRegression
import numpy as np
import math
import calendar
import certifi
from time import strptime
from io import StringIO
import io
import os
from datetime import timedelta
from exstracs.exstracs_timer import Timer
from exstracs.exstracs_configparser import ConfigParser
from exstracs.exstracs_offlineenv import Offline_Environment
from exstracs.exstracs_onlineenv import Online_Environment
from exstracs.exstracs_algorithm import ExSTraCS
from exstracs.exstracs_constants import *
from exstracs.exstracs_at import AttributeTracking
from exstracs.exstracs_ek import ExpertKnowledge

app=Flask(__name__)

@app.route('/traffic', methods=['POST'])
def upload_traffic_file():
    begin = time.perf_counter()
    # checking if the file is present or not.
    if 'file' not in request.files:
        return "No file found"
    file = request.files['file']
    file.save("datalake/"+file.filename)
    end = time.perf_counter() - begin
    return render_template('home.html', data2= str(round(end,2)), data3="Traffic file successfuly saved to datalake")

@app.route('/match', methods=['POST'])
def webscrap_post():
    begin = time.perf_counter()
    args=request.form['year']
    tables = pd.read_html("http://www.worldfootball.net/teams/manchester-united/"+args+"/3/")
    df=tables[1]
    df.columns=['Round','date','time','place','logo','Opponent','Results','desc']
    df=df.drop(df.columns[[4,7]], axis=1)
    df = df.dropna(how='any',axis=0) 
    df.to_csv("datalake/"+args+"matches.csv", index = False, sep=',', encoding='utf-8')
    end = time.perf_counter() - begin
    print("match file succesfully saved to datalake") 
    return render_template('home.html', data2= str(round(end,2)), data3="Match file successfuly saved to datalake") 

@app.route('/integration/<monthfrom>/<yearfrom>/<monthto>/<yearto>/<place>/<int:bfrduration>/<int:afterduration>')
def get_matches_years(monthfrom,yearfrom,monthto,yearto,place,bfrduration,afterduration):
    begin = time.perf_counter()
    years=int(yearto)+2
    mnthfrom=strptime(monthfrom, '%b').tm_mon
    mnthto=strptime(monthto, '%b').tm_mon
    print('processing data matches....')
    # get data matches by year
    df=pd.DataFrame()
    for year in range(int(yearfrom),years):
        dftem=pd.pandas.read_csv('./datalake/'+str(year)+'matches.csv', index_col=False, header=0)
        df=df.append(dftem) 

    df['date'] = pd.to_datetime(df['date']+' '+df['time'],format='%d/%m/%Y %H:%M')
    df.index=df['date'] 
    print('filtering matches based on input range...')
    
    df=df[(df['date'].apply(lambda x:x.strftime('%Y-%m'))>=yearfrom+'-'+str(mnthfrom)) & (df['date'].apply(lambda x:x.strftime('%Y-%m'))<=yearto+'-'+str(monthto))]
    if place == "home":
        df=df[df['place'].str.match("H")]
    elif place == "away":
        df=df[df['place'].str.match("A")]
    else:
        pass
    print(df)

    df['DayNumber']=pd.DatetimeIndex(df['date']).weekday
    df['DayName']=pd.DatetimeIndex(df['date']).weekday_name
    df['Week']=np.where(df['DayNumber']<=4, "Weekdays", "Weekend")

    df.to_csv("datalake/"+monthfrom+yearfrom+monthto+yearto+"integrationmatch.csv", index = False, sep=',', encoding='utf-8')
    print('processing traffic data ....')   
    # get traffic data 
    dftraffic=pd.DataFrame()
    for year in range(int(yearfrom),years):
        for mnt in range(mnthfrom,mnthto+1):
            month=calendar.month_abbr[mnt]
            if os.path.isfile('./datalake/'+month+str(year)+'.csv'):
                dftemp=pd.pandas.read_csv('./datalake/'+month+str(year)+'.csv',index_col=False, header=0)
                dftemp['Sdate'] = pd.to_datetime(dftemp['Sdate'],format='%d/%m/%Y %H:%M')
                dftraffic=dftraffic.append(dftemp)
            else:
                pass
                
        # reset month to iterate 
        mnthfrom=1
    dftraffic['Outlier']=0
    
    ### data preparation
    print('processing data preparation')
    # remove row when volume==0 and averageSpeed!=0
    dftraffic=dftraffic[~((dftraffic['Volume']==0) & (dftraffic['AvgSpeed']!=0))]
    dftraffic=dftraffic[~(dftraffic['LaneNumber']==7)]
    # change inconsistent lanenumber
    dftraffic.loc[(dftraffic['LaneNumber']==4) & (dftraffic['DirectionDescription']=="North") & (dftraffic['LaneDirection']==1),'LaneDirection']=2 
    dftraffic.loc[(dftraffic['LaneNumber']==4) & (dftraffic['DirectionDescription']=="North") & (dftraffic['LaneDirection']==2),'DirectionDescription']="South"
    # extract date & hour from Sdate
    dftraffic['Date']=pd.DatetimeIndex(dftraffic['Sdate']).strftime('%Y/%m/%d')
    dftraffic['Hour']=pd.DatetimeIndex(dftraffic['Sdate']).strftime('%H')
    # extract daynumber (Monday=0 and Sunday=6), dayname (Monday to Sunday) and weekdays/weekend
    dftraffic['DayNumber']=pd.DatetimeIndex(dftraffic['Sdate']).weekday
    dftraffic['DayName']=pd.DatetimeIndex(dftraffic['Sdate']).weekday_name
    dftraffic['Week']=np.where(dftraffic['DayNumber']<=4, "Weekdays", "Weekend")

    print('processing outlier prediction...')
    # set outlier to 1 where traffic occured on home match 
    for row in df.itertuples():
        kickoff=row[0].strftime("%Y-%m-%d %H:00")
        timebfr=(row[0]-timedelta(hours=bfrduration)) 
        timebefore=timebfr.strftime("%Y-%m-%d %H:00")
        timeafr=row[0] +timedelta(minutes=105)
        timeafter=timeafr.strftime("%Y-%m-%d %H:00")
        dftraffic.loc[(dftraffic['DirectionDescription']=="North") & (dftraffic['Sdate']>=timebefore) & (dftraffic['Sdate']<=row[0])  ,'Outlier']=1
        dftraffic.loc[(dftraffic['DirectionDescription']=="South") & (dftraffic['Sdate']>=timeafter) & (dftraffic['Sdate']<=row[0]+timedelta(hours=afterduration,minutes=105)) ,'Outlier']=1

    buffer=StringIO()
    print('saving file '+monthfrom+yearfrom+monthto+yearto+"trafficmatch.csv" +' to datalake.. ')
    dftraffic.to_csv("datalake/"+monthfrom+yearfrom+monthto+yearto+"trafficmatch.csv", index = False, sep=',', encoding='utf-8')
    dftraffic.to_csv(buffer,encoding='utf-8',index=False)
    end = time.perf_counter() - begin
    print('respond to browser..')
    print('Time processing : ' + str(round(end,2)) + ' Seconds')
    # return Response(buffer.getvalue(), mimetype='text/csv')
    return render_template('home.html', data2= str(round(end,2)), data3=buffer.getvalue()) 

#get traffic from monthyear
# @app.route('/arima/<day>/<int:lanedirection>/<int:hour>/<int:hour2>', methods=['GET'])
# def get_arima(day,lanedirection,hour, hour2):
##@app.route('/arima/<day>/<int:lanenumber>/<int:hour>/<int:hour2>', methods=['GET'])
##def get_arima(day,lanenumber,hour, hour2):
@app.route('/arima/<weekday>/<int:lanedirection>/<int:hour>/<int:hour2>', methods=['GET'])
def get_arima(weekday,lanedirection,hour, hour2):
##@app.route('/arima/<day>/<int:lanenumber>/<int:hour>', methods=['GET'])
##def get_arima(day,lanenumber,hour):
##@app.route('/arima/<day>/<int:hour>/<int:hour2>', methods=['GET'])
##def get_arima(day, hour, hour2):  
##@app.route('/arima/<day>/<int:lanedirection>/<int:hour>', methods=['GET'])
##def get_arima(day, lanedirection, hour): 
##@app.route('/arima/<day>/<int:hour>', methods=['GET'])
##def get_arima(day, hour): 
    begin = time.perf_counter()
    dateparse = lambda dates: datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    series = read_csv('./datalake/Nov2012Dec2017trafficmatch.csv', parse_dates=['Sdate'], index_col='Sdate',date_parser=dateparse)
   
    # dat = series.loc[(series['DayName'] == day) & (series['LaneDirection'] == lanedirection) & (series['Hour'] >= hour) & (series['Hour'] <= hour2)]
    ##dat = series.loc[(series['DayName'] == day) & (series['LaneNumber'] == lanenumber) & (series['Hour'] >= hour) & (series['Hour'] <= hour2)]
    dat = series.loc[(series['Week'] == weekday) & (series['LaneDirection'] == lanedirection) & (series['Hour'] >= hour) & (series['Hour'] <= hour2)]
    ##dat = series.loc[(series['DayName'] == day) & (series['Hour'] >= hour) & (series['Hour'] <= hour2)]
    ##dat = series.loc[(series['DayName'] == day) & (series['LaneDirection'] == lanedirection) & (series['Hour'] == hour)]
    ##dat = series.loc[(series['DayName'] == day) & (series['Hour'] == hour)]
    
    
    ##train = dat.loc[(dat['Date'] != '28/11/2012') & (dat['Date'] != '05/12/2012') & (dat['Date'] != '26/12/2012') & (dat['Date'] != '04/12/2013')]
    ##test = dat.loc[(dat['Date'] != '05/12/2012') & (dat['Date'] != '26/12/2012') & (dat['Date'] != '04/12/2013')]
    ##train = dat.loc[(dat['Date'] != '10/12/2013') & (dat['Date'] != '02/12/2014')]
    ##test = dat.loc[(dat['Date'] != '02/12/2014')]
    train = dat.loc[(dat['Outlier'] ==0)]
    test = dat

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

    df=test[['DayName', 'LaneNumber', 'DirectionDescription', 'Volume', 'AvgSpeed', 'Outlier']]
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
    print(df2)

    #BalancedAccuracy = balanced_accuracy_score(df['Outlier'], df['OutlierPrediction'])
    tn, fp, fn, tp = confusion_matrix(df['Outlier'].values, df['OutlierPrediction'].values).ravel()
    sensitivity=tp/(tp+fn)
    #sensitivity=recall_score(df['Outlier'], df['OutlierPrediction'], average='weighted')
    specificity=tn/(fp+tn)
    BalancedAccuracy=(sensitivity+specificity)/2
    #FPRate = fp/(fp+tn)
    end = time.perf_counter() - begin
    print('Balanced Accuracy=%.2f' % (BalancedAccuracy))

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
    # return render_template("home.html", graph=rsp, data=df2.to_html(),baccuracy=BalancedAccuracy)
    return render_template("home.html", graph=rsp,data="Balanced Accuracy = "+str(round(BalancedAccuracy,2)), data3=df2.to_html(), data2=round(end,2))

@app.route('/lstmrnn/<inputfile>/<weekday>/<int:lanedirection>/<int:hourfrom>/<int:hourto>', methods=['GET'])
def lstmnn(inputfile,weekday,lanedirection,hourfrom,hourto):
# @app.route('/lstmrnn/<inputfile>/<day>/<int:lanedirection>/<int:hourfrom>/<int:hourto>', methods=['GET'])
# def lstmnn(inputfile,day,lanedirection,hourfrom,hourto):
    begin = time.perf_counter()
    data_dir_path = './datalake'
    model_dir_path = './models'
    ##df = pd.read_csv(data_dir_path + '/Nov2012Dec2014trafficmatches.csv')
    # df = pd.read_csv(data_dir_path + '/Nov2012Dec2017trafficmatch.csv')
    df = pd.read_csv(data_dir_path + '/'+inputfile)
    ##print(df.head())
    dat=df.loc[(df['Week']==weekday) & (df['LaneDirection']==lanedirection) & (df['Hour'] >= hourfrom) & (df['Hour'] <= hourto)]
    # dat=df.loc[(df['DayName']==day) & (df['LaneDirection']==lanedirection) & (df['Hour'] >= hourfrom) & (df['Hour'] <= hourto)]
    # dat=df.loc[(df['Week']=='Weekdays') & (df['DirectionDescription']=='South') & (df['Hour'] >= 21) & (df['Hour'] <= 23)]
    ## dat=df.loc[(df['Week']==weekday) & (df['DirectionDescription']==direction) & (df['Hour'] >= hourfrom) & (df['Hour'] <= hourto)]
    dat.insert(0, 'Row', range(0, 0+len(dat)))
    dat=dat[['Row', 'Sdate', 'DayName', 'LaneNumber', 'DirectionDescription', 'Volume', 'AvgSpeed', 'Outlier']]
    print(dat)
    ##traffic_data = pd.read_csv(data_dir_path + '/test_data.csv', header=None)
    ##traffic_data = pd.read_csv(data_dir_path + '/test_south.csv', header=None)
    traffic_data = dat[['Volume']]
    print(traffic_data.head())
    traffic_np_data = traffic_data.values
    scaler = MinMaxScaler()
    traffic_np_data = scaler.fit_transform(traffic_np_data)
    print(traffic_np_data.shape)

    ae = LstmAutoEncoder()

    # fit the data and save model into model_dir_path
    ae.fit(traffic_np_data[:, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)

    # load back the model saved in model_dir_path detect anomaly
    ae.load_model(model_dir_path)
    anomaly_information = ae.anomaly(traffic_np_data)
    reconstruction_error = []
   
    # new dataframe to store idk and anomaly 
    colnames =  ['Row', 'OutlierPrediction']
    df2  = pd.DataFrame(columns = colnames)  

    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
        df2.loc[len(df2)] = [idx, ('abnormal' if is_anomaly else 'normal')]
        #if is_anomaly :
            #df2.loc[len(df2)] = [idx, 'abnormal']
        #else:
            #pass
        reconstruction_error.append(dist)
    #print(df2)
    
    dat['OutlierPrediction']=np.where(df2['OutlierPrediction']=='abnormal', 1, 0)
    df3=dat.loc[dat['OutlierPrediction'] == 1]
    print(df3)
    
    tn, fp, fn, tp = confusion_matrix(dat['Outlier'].values, dat['OutlierPrediction'].values).ravel()
    sensitivity=tp/(tp+fn)
    #sensitivity=recall_score(df['Outlier'], df['OutlierPrediction'], average='weighted')
    specificity=tn/(fp+tn)
    BalancedAccuracy=(sensitivity+specificity)/2
    #FPRate = fp/(fp+tn)
    end = time.perf_counter() - begin
    print('Balanced Accuracy=%.2f' % (BalancedAccuracy))

    # visualize_reconstruction_error(reconstruction_error, ae.threshold)

     #plot
    img=io.BytesIO()
    pyplot.plot(reconstruction_error, marker='o', ms=3.5, linestyle='',
             label='Point')
    pyplot.hlines(ae.threshold, xmin=0, xmax=len(reconstruction_error)-1, colors="r", zorder=100, label='Threshold')
    pyplot.legend()
    pyplot.ylabel("Dist")
    pyplot.xlabel("Data point index")
    #pyplot.show()
    pyplot.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()
    rsp='<img src="data:image/png;base64,{}">'.format(plot_url)
    
    # return rsp
    # return render_template("home.html", graph=rsp, data=df3.to_html(),baccuracy=BalancedAccuracy)
    return render_template("home.html", graph=rsp,data="Balanced Accuracy = "+str(round(BalancedAccuracy,2)), data3=df3.to_html(), data2=round(end,2))

    # return 'lstm processed..'


@app.route('/exstracs', methods=['POST'])
def exstracs_file():
    # start counter 
    begin = time.perf_counter()
    
    # global baccuracy
    # checking if the file is present or not.
    if 'trainfile' not in request.files:
        return "No file found"
    
    trainfile = request.files['trainfile']
    # trainfile2 = request.files['trainfile']
    trainfile.save("datalake/training.txt")
    # trainfile2.save("exstracs/Datasets/training.txt")
    # trainfile.save("exstracs/Datasets/training.txt")

    testfile = request.files['testfile']
    # testfile2 = request.files['testfile']
    testfile.save("datalake/testing.txt")
    # testfile2.save("exstracs/Datasets/testing.txt")
    # testfile.save("exstracs/Datasets/testing.txt")

    helpstr = """
    Failed attempt to run ExSTraCS.
    """
    #Obtain path to configuration file
    # configurationFile = "ExSTraCS_Configuration_File_Complete.txt"#"ExSTraCS_Configuration_File_Minimum.txt"#"ExSTraCS_Configuration_File_Complete.txt"
    # configurationFile = "ExSTraCS_Configuration_File_Minimum.txt"#"ExSTraCS_Configuration_File_Complete.txt"
    configurationFile = "ExSTraCS_Configuration_File_Recommended.txt"#"ExSTraCS_Configuration_File_Complete.txt"

    #Initialize the Parameters object - this will parse the configuration file and store all constants and parameters.
    ConfigParser(configurationFile)
    if cons.offlineData:  
        print('ExSTraCS Offline Environment Mode Initiated.')
        if cons.internalCrossValidation == 0 or cons.internalCrossValidation == 1:  #No internal Cross Validation
            #Engage Timer - tracks run time of algorithm and it's components.
            timer = Timer() #TIME
            cons.referenceTimer(timer)
            cons.timer.startTimeInit()
            #Initialize the Environment object - this manages the data presented to ExSTraCS 
            env = Offline_Environment()
            cons.referenceEnv(env) #Send reference to environment object to constants - to access from anywhere in ExSTraCS
            cons.parseIterations() 
            
            #Instantiate ExSTraCS Algorithm
            algorithm = ExSTraCS()
            if cons.onlyTest:
                cons.timer.stopTimeInit()
                algorithm.runTestonly()
            else:
                if cons.onlyRC:
                    cons.timer.stopTimeInit()
                    algorithm.runRConly()
                else: 
                    if cons.onlyEKScores:
                        cons.timer.stopTimeInit()
                        EK = ExpertKnowledge(cons)
                        print("Algorithm Run Complete")
                    else: #Run the ExSTraCS algorithm.
                        if cons.useExpertKnowledge: #Transform EK scores into probabilities weights for covering. Done once. EK must be externally provided.
                            cons.timer.startTimeEK()
                            EK = ExpertKnowledge(cons)
                            cons.referenceExpertKnowledge(EK)
                            cons.timer.stopTimeEK()
                            
                        if cons.doAttributeTracking:
                            cons.timer.startTimeAT()
                            AT = AttributeTracking(True)
                            cons.timer.stopTimeAT()
                        else:
                            AT = AttributeTracking(False)
                        cons.referenceAttributeTracking(AT)
                        cons.timer.stopTimeInit()
                        algorithm.runExSTraCS()
        else:
            print("Running ExSTraCS with Internal Cross Validation") 
            for part in range(cons.internalCrossValidation):
                cons.updateFileNames(part)  
                
                #Engage Timer - tracks run time of algorithm and it's components.
                timer = Timer() #TIME
                cons.referenceTimer(timer)
                cons.timer.startTimeInit()
                #Initialize the Environment object - this manages the data presented to ExSTraCS 
                env = Offline_Environment()
                cons.referenceEnv(env) #Send reference to environment object to constants - to access from anywhere in ExSTraCS
                cons.parseIterations() 
                
                #Instantiate ExSTraCS Algorithm
                algorithm = ExSTraCS()
                if cons.onlyTest:
                    cons.timer.stopTimeInit()
                    algorithm.runTestonly()
                else:
                    if cons.onlyRC:
                        cons.timer.stopTimeInit()
                        algorithm.runRConly()
                    else: 
                        if cons.onlyEKScores:
                            cons.timer.stopTimeInit()
                            cons.runFilter()
                            print("Algorithm Run Complete") 
                        else: #Run the ExSTraCS algorithm.
                            if cons.useExpertKnowledge: #Transform EK scores into probabilities weights for covering. Done once. EK must be externally provided.
                                cons.timer.startTimeEK()
                                EK = ExpertKnowledge(cons)
                                cons.referenceExpertKnowledge(EK)
                                cons.timer.stopTimeEK()
                                
                            if cons.doAttributeTracking:
                                cons.timer.startTimeAT()
                                AT = AttributeTracking(True)
                                cons.timer.stopTimeAT()
                            else:
                                AT = AttributeTracking(False)
                            cons.referenceAttributeTracking(AT)
                            cons.timer.stopTimeInit()
                            algorithm.runExSTraCS()
    else: #Online Dataset (Does not allow Expert Knowledge, Attribute Tracking, Attribute Feedback, or cross-validation)
        #Engage Timer - tracks run time of algorithm and it's components.
        print("ExSTraCS Online Environment Mode Initiated.") 
        timer = Timer() #TIME
        cons.referenceTimer(timer)
        cons.timer.startTimeInit()
        cons.overrideParameters()
        
        #Initialize the Environment object - this manages the data presented to ExSTraCS 
        env = Online_Environment()
        cons.referenceEnv(env) #Send reference to environment object to constants - to access from anywhere in ExSTraCS
        cons.parseIterations() 
        
        #Instantiate ExSTraCS Algorithm
        algorithm = ExSTraCS()
        cons.timer.stopTimeInit()
        if cons.onlyRC:
            algorithm.runRConly()
        else: 
            algorithm.runExSTraCS()

    # read from file txt
    df = pd.read_csv('./Local_Output/training_ExSTraCS_5000_RulePop.txt', sep="\t", header=None)
    df.columns = ['Specified','Condition','Phenotype','Fitness','Accuracy','Numerosity','AveMatchSetSize','TimeStampGA','InitTimeStamp','Specificity','DeletionProb','CorrectCount','MatchCount','CorrectCover','MatchCover','EpochComplete']
    df1 = df.iloc[:,0:3]
    df2=df1.loc[df1['Phenotype'] == '1']
    
    file=open("./Local_Output/balancedAccuracy.txt","r")
    dfba=file.read()
    end = time.perf_counter() - begin

    # dfba = pd.read_csv('./Local_Output/balancedAccuracy.txt', sep="\t", header=None)

    return render_template("home.html",data3=df2.to_html(), data=dfba, data2=round(end,2))

    # return "training and testing files successfully saved"

@app.after_request
def add_header(r):
  
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

app.run(port=5000)