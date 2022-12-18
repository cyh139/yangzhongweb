#  streamlit run c:\Users\cyh\Desktop\web\pre_app3.py [ARGUMENTS]
#  Local URL: http://localhost:8501
#  Network URL: http://192.168.155.46:8501

import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
from plotly import graph_objs as go
from tk import *
from math import log, exp
import pmdarima as pm
from statsmodels.tsa.seasonal import STL

st.title("扬中负荷预测工具")

# 上传数据
uploaded_file = st.file_uploader('',type=['csv'])

#下载数据
def data_download(csv):
    csv.to_csv()
    csv = bytes(str(csv),encoding='utf-8')
    st.download_button(label="Download Predicted Data",data=csv,
        file_name='pre_data.csv',mime='text/csv')

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[selected_stocks1],y=data[selected_stocks2],name=''))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_pre_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pre_index,y=pre_data[selected_pre_stocks2],name=''))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

#负荷预测代码
def function_stl(data, nOut):
    data = data.iloc[:, 1]
    stl = STL(data, period=12)
    res = stl.fit()

    smodel = pm.auto_arima(res.trend, start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, seasonal=False,
                           d=None, trace=False,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
    trend = smodel.predict(n_periods=nOut, return_conf_int=False)

    res_seasonal = np.array(res.seasonal)
    nums = len(res_seasonal)
    if nums < 12:
        raise Exception("Input data must be longer than 12, now is", nums)
    s = []
    for i in range(12):
        temp, k = [], 0
        while((i+1+12*k)<=nums):
            temp.append(res_seasonal[-(i+1+12*k)])
            k += 1
        if len(temp) >= 10:
            temp = temp[:10]
        weight = [1.5 ** i for i in range(len(temp), 0, -1)]
        s.append(np.average(temp, weights=weight))
    s = np.array(s[::-1])
    seasonal = np.concatenate([s]*(int(nOut / 12) + 1))[:nOut]

    return trend + seasonal

def function_sarima(data, nOut):
    # ARIMA
    data = data.iloc[:, 1]
    # data = np.log(data)
    smodel = pm.auto_arima(data, start_p=0, start_q=0,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=False,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
    fitted, confint = smodel.predict(n_periods=nOut, return_conf_int=True)
    # fitted = np.exp(fitted)

    return fitted

def liner_fitting(data_x, data_y):
    x_a, y_a = np.mean(data_x), np.mean(data_y)
    return_k = sum([y_i * x_i - y_a * x_i for x_i, y_i in zip(data_x, data_y)]) / \
               sum([x_i * (x_i - x_a) for x_i in data_x])
    return_b = y_a - return_k * x_a
    return [return_k, return_b]

def function_lf(data, nOut):
    data = list(np.array(data.iloc[:, 1]))
    time = [i for i in range(len(data))]
    value = [log(data[i]) for i in range(len(data))]

    k, b = liner_fitting(time, value)

    fitted = [exp(b + k * j) for j in range(len(data), len(data) + nOut)]

    return fitted

def function_gf(data, nOut):
    d = np.array(data.iloc[:, 1]).reshape(1, -1)
    d_cum = np.cumsum(d, axis=1)

    C = (d_cum[:, 1:] + d_cum[:, :-1]) / 2
    E = np.concatenate([-C, np.ones_like(C)], axis=0)
    c = np.dot(np.dot(np.linalg.inv(np.dot(E, E.T)), E), d[:, 1:].T)

    F = (d[0, 0] - c[1] / c[0]) * np.exp(-c[0] * np.arange(data.shape[0] + nOut)) + c[1] / c[0]
    G = np.diff(F)

    fitted = np.concatenate([[d[0, 0]], G])
    fitted = fitted[d.size:]
    return fitted

def pre(df,pre_obj,timescale,nOut,method):
    cols = df.columns
    obj_num_ = 0
    for i in cols:
        obj_num_ = obj_num_+1
        if i == pre_obj:
            obj_num = obj_num_
    date = pd.to_datetime(df.iloc[:, 0])
    if timescale == 'year':
        fitted = np.full((nOut), np.nan)
        if method == 'LF':
            fitted[:] = function_lf(df.iloc[:,[0,obj_num]], nOut)
        if method == 'GF':
            fitted[:] = function_gf(df.iloc[:,[0,obj_num]], nOut)
        index_f = pd.date_range(date.iloc[-1] + pd.DateOffset(years=1), periods=nOut, freq='YS')
        st.write(' predict is done!')
    if timescale == 'month':
        fitted = np.full((nOut), np.nan)
        if method == 'STL':
            fitted[:] = np.array(function_stl(df.iloc[:,[0,obj_num]], nOut))
        if method == 'SARIMA':
            fitted[:] = np.array(function_sarima(df.iloc[:,[0,obj_num]], nOut))
        st.write(' predict is done!')
        index_f = pd.date_range(date.iloc[-1] + pd.DateOffset(months=1), periods=nOut, freq='MS')
    params = {'minute': [
                        [i+1 for i in range(16)],  # n_out
                        3,                         # n_in1 日内使用的前n个时刻数据
                        [1, 2, 3],                    # n_in2 日前使用天的数据
                        48,                        # n_split 每组点数
                        [0]],                   # 会重复使用的内容
                'day': [
                        [1],                              # n_out 提前时间
                        [1, 2],                           # n_out + this的使用数据
                        48,                               # n_split 每组点数
                        [0]                            # 会重复使用的内容
                      ]}
    if timescale == 'day':
        fitted = np.full((nOut), np.nan)
        if method == 'STL':
            fitted[:] = np.array(function_stl(df.iloc[:,[0,obj_num]], nOut))
        st.write(' predict is done!')
        index_f = pd.date_range(date.iloc[-1] + pd.DateOffset(months=1), periods=nOut, freq='D')
    if timescale == 'minute': 
        step = int((date.iloc[-1] - date.iloc[-2]) / pd.to_timedelta('0:1:0'))
        freq = str(step) + 'T'
        fitted = np.full((nOut), np.nan)
        if method == 'STL':
            fitted[:] = np.array(function_stl(df.iloc[:,[0,obj_num]], nOut))
        st.write(' predict is done!')
        index_f = pd.date_range(date.iloc[-1] + pd.DateOffset(months=1), periods=nOut, freq=freq)
    
    #index_i = 0
    #fitted_ = np.full((nOut,2), np.nan)
    #fitted_[:,1] = np.array(fitted)
    #for i in index_f:
    #    fitted_[index_i,0] = i
    #    index_i = index_i+1
    #columns_f = ["Time","pre_data"]
    #df_ = pd.DataFrame(fitted_,columns=columns_f)
    columns_f = ["pre_data"]
    df_ = pd.DataFrame(fitted, columns=columns_f,index=index_f)
    return df_,df_.index

if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    
    st.subheader("Raw data")
    st.write(data)
    stocks = data.columns
    pre_obj = st.selectbox("Select object for prediction",stocks)
    selected_stocks1 = st.selectbox("Select xaxis for prediction",stocks)
    selected_stocks2 = st.selectbox("Select yaxis for prediction",stocks)
    
    plot_raw_data()

    # 时间尺度
    # timescale = 'month'
    # method = 'SARIMA'     # 季节时间序列
    # method = 'STL'        # 时间序列分解

    # timescale = 'year'
    # method = 'LF'         # 回归预测
    # method = 'GF'         # 灰度预测
    timescales = ("year","month","day","minute")
    timescale_ = st.selectbox("Select timescale for prediction",timescales)
    if timescale_ == "year":
        methods = ("LF","GF")
    elif timescale_ == "month":
        methods = ("SARIMA","STL")
    else:
        methods = ("STL","STL")
    method_ = st.selectbox("Select method for prediction",methods)
    nOut_ = st.slider("nOut of prediction:",1,20)
    
    pre_data,pre_index = pre(data,pre_obj,timescale_,nOut_,method_)
    
    pre_data = pd.DataFrame(pre_data)
    st.write(type(pre_data))
    st.subheader("Pre data")
    st.write(pre_data)
    pre_stocks = pre_data.columns
    #selected_pre_stocks1 = st.selectbox("Select xaxis for prediction",pre_stocks)
    selected_pre_stocks2 = st.selectbox("Select yaxis for prediction",pre_stocks)
    plot_pre_data()
    
    data_download(pre_data)
