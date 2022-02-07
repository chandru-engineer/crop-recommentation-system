from django.shortcuts import render
import pandas as pd
import numpy as np
import csv
import joblib

import pandas as pd
import numpy
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly
import  random

df = pd.read_csv('data.csv')

colorarr = ['#0592D0','#Cd7f32', '#E97451', '#Bdb76b', '#954535', '#C2b280', '#808000','#C2b280', '#E4d008', '#9acd32', '#Eedc82', '#E4d96f',
           '#32cd32','#39ff14','#00ff7f', '#008080', '#36454f', '#F88379', '#Ff4500', '#Ffb347', '#A94064', '#E75480', '#Ffb6c1', '#E5e4e2',
           '#Faf0e6', '#8c92ac', '#Dbd7d2','#A7a6ba', '#B38b6d']
classes = [
            'apple', 'banana', 'blackgram', 'chickpea',
            'coconut', 'coffee',
            'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
            'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
            'pigeonpeas', 'pomegranate', 'rice', 'watermelon']


df = pd.read_csv("data.csv")

def intractive_plot(df, feature, name):
    """
    This Function helps to create intractive Chart
    ATTRIBUTE:
    df: original DataFrame
    feature: which column need to be
    name : feature name
    """

    df_label = pd.pivot_table(df, index=['label'], aggfunc='mean')
    df_label_feature = df_label.sort_values(by=feature, ascending = False)

    fig = make_subplots(rows = 1, cols = 2)

    top = {

        'y': df_label_feature[feature][:10].sort_values().index,
        'x': df_label_feature[feature][:10].sort_values()
    }
    last = {

        'y': df_label_feature[feature][-10:].sort_values().index,
        'x': df_label_feature[feature][-10:].sort_values()
    }


    fig.add_trace(
        go.Bar(top,
               name='Highest {} Needed'.format(name),
               marker_color = random.choice(colorarr),
               orientation = 'h',
               text = top['x']
              ),
        row = 1, col = 1
    )
    fig.add_trace(
        go.Bar(last,
               name='Least {} Needed'.format(name),
               marker_color = random.choice(colorarr),
               orientation = 'h',
               text = top['x']
              ),
        row = 1, col = 2
    )

    fig.update_traces(texttemplate = '%{text}', textposition = 'inside')
    fig.update_layout(title_text = name,
                      plot_bgcolor = 'white',
                      font_size = 12,
                      font_color = 'black',
                      height = 500
                     )


    fig.update_xaxes(showgrid = False)
    fig.update_yaxes(showgrid = False)
    plt_div = plotly.offline.plot(fig, output_type='div')
    return plt_div

# Create your views here.
def home(request):
    return render(request, 'app/home.html')

def usecase(request):
    return render(request, 'app/usecase.html')

def analysis(request):

    graph_1 = intractive_plot(df, feature = 'N', name = 'Ratio of Nitrogen')
    graph_2 = intractive_plot(df, feature = 'P', name = ' Radio of Phosphorous')
    graph_3 = intractive_plot(df, feature = 'K', name = 'Radio of Potassium')
    graph_4 = intractive_plot(df, feature = 'temperature', name = ' Ratio of Temperature')
    graph_5 = intractive_plot(df, feature = 'humidity', name = 'Ratio of Humidity')
    graph_6 = intractive_plot(df, feature = 'ph', name = 'Ratio of ph')
    graph_7 = intractive_plot(df, feature = 'rainfall', name = 'Ratio of Rainfall')

    graph_list = [graph_1, graph_2,  graph_3,  graph_4,  graph_5,  graph_6,  graph_7]

    context = {'graph': graph_list}
    return render(request, 'app/analysis.html', context)

def predict(request):

    model = joblib.load('lightgbm.pkl')
    list_ = []
    list_.append(int(request.GET['N']))
    list_.append(int(request.GET['P']))
    list_.append(int(request.GET['K']))
    list_.append(int(request.GET['temperature']))
    list_.append(int(request.GET['humidity']))
    list_.append(int(request.GET['ph']))
    list_.append(int(request.GET['rainfall']))

    answer = model.predict([list_])
    predict_pro = model.predict_proba([list_])
    list_proba = []
    for i in [-2, -3, -4, -5]:
        list_proba.append(classes[np.argsort(np.max(predict_pro, axis=0))[i]])
    print(list_proba)
    return render(request, 'app/predict.html', {'answer': answer[0], 'list_':list_proba})
