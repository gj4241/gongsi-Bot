
# coding: utf-8

# In[1]:


import dart_api
import finance_statements as fs
import naver_stock
import candle_chart
import make_text_image as mti
import datetime
from xml.etree.ElementTree import *
from time import sleep
from collections import OrderedDict
import pandas as pd
from collections import Counter
import sys
import os
import re
import uuid
import cgi
from time import sleep
import subprocess
import ftplib

import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from email.mime.application import MIMEApplication


# In[2]:


def make_object(dart_df, no):
    crp_cd, crp_cls, crp_nm, rcp_dt, rcp_dt, rcp_no, rpt_nm, period = (dart_df['crp_cd'][no], dart_df['crp_cls'][no], dart_df['crp_nm'][no]
                                                  , dart_df['rcp_dt'][no], dart_df['rcp_dt'][no], dart_df['rcp_no'][no]
                                                  , dart_df['rpt_nm'][no], dart_df['period'][no])
    
    return crp_cd, crp_cls, crp_nm, rcp_dt, rcp_dt, rcp_no, rpt_nm, period


# In[3]:


def make_xml(title_text, total_text, now):
    article=Element('article')
    regitime=Element('regitime')

    ##날짜##
    regitime.text = now.strftime('%Y-%m-%d %H:%M')

    article.append(regitime)
    ##title##
    SubElement(article,'title').text = title_text
    ##body##
    SubElement(article,'body').text = total_text

    dump(article)
    ElementTree(article).write('xml/' + rcp_no + '.xml')


# In[411]:


loginid = 'm.robo.walt@gmail.com'
loginpw = '***'

sender = 'm.robo.walt@gmail.com'
reciplist = ['gj4241@gmail.com']


# In[6]:


crp_list = pd.read_csv('crp_list.csv',encoding='utf-8',dtype=str)
crp_list.head()


# In[7]:


report_df = pd.read_csv('2018년사업보고서.csv',encoding='utf-8',dtype='str')
report_df = report_df.drop('Unnamed: 0',axis=1)
report_df = report_df.drop('index',axis=1)
report_df = report_df.drop_duplicates()
report_df = report_df.reset_index()


# In[8]:


report_df.head(10)


# In[9]:




import dart_api
import finance_statements as fs
import re
import pandas as pd
import hgtk
import math
import numpy as np
from PIL import Image
import os
import datetime
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib 
import matplotlib.ticker as ticker
krfont = {'family' : 'nanumgothic', 'weight' : 'bold', 'size'   : 10}
matplotlib.rc('font', **krfont)
def money_scaler(money): # 돈단위 함수    
    if money > 0:
        money = str(money)
        if 8 >= len(money) > 4:
            money = '이익 ' + money[:-4] + '만원'
        elif 12 >= len(money) > 8:
            money = '이익 ' + str(int(money[:-8])) + '억원'
        elif len(money) > 12:
            money = '이익 ' + money[:-12] + '조 ' + str(int(money[-12:-8])) + '억원'
        else :
            money = '이익 ' + money + '원'
    else :
        money = str(money)
        money = money.replace('-', '')
        if 8>= len(money) > 4:
            money = '손실 ' + money[:-4] + '만원'
        elif 12 >= len(money) > 8:
            money = '손실 ' + money[:-8] + '억원'
        elif len(money) > 12:
            money = '손실 ' + money[:-12] + '조 ' + str(int(money[-12:-8])) + '억원'
        else :
            money = '손실 ' + money + '원'
            
    return money


# In[4]:


def crease(x):
    if x >= 0:
        a = " 증가"
    elif x < 0:
        a = " 감소"        
    return a
def joinLists(listList):
    '''Convert a list of which elements are lists into a single list

    Keyword arguments :
    list -- each element is list
    
    Returns: list
    
    ex)
    input : [['A','B'],['C','D','E']]
    output : ['A','B','C','D','E']

    '''
    result = []
    for i in range(0, len(listList)):
        result = result + listList[i]
    return result


# In[244]:


def plot_bar_sales(df_column3_row2, title_text, rcp_no, no):
    #my_font_title = fm.FontProperties(fname="./baeminfont.ttf", size = 14)
    #my_font_label = fm.FontProperties(fname="./baeminfont.ttf", size = 12) #폰트 설정하는 것. 한귿 들어가는 데마다 이거써줘야 함
    
    fig, ax = plt.subplots(figsize=(10,6))


    '''INPUT : pandas dataframe of 2 rows and 3 columns
    
    The index of the input must be in type of 'datetime'.
    The values must be numeric.
    
    OUTPUT : 3 pairs, bars grouped by 2. each pair indicates purchase volume

    '''

    # 타이틀
    plt.title('%s'%title_text, size = 16)
    
    # 막대의 가로 위치 조정을 위한 변수 세팅
    barWidth = 0.25

    loc_latest = np.arange(len(df_column3_row2.columns))
    loc_previous = [x + barWidth + 0.06 for x in loc_latest] # 0.03을 더해주는 이유는 같은 그룹 내 막대 간의 gap을 위해서
    loc_pprevious = [x + barWidth + 0.06 for x in loc_previous]
    if len(df_column3_row2.index) ==3:
        locAll_axis_x = [*loc_latest, *loc_previous ,*loc_pprevious] #막대들이 위치할 x 좌표
        locAll_axis_y = joinLists(df_column3_row2.values.tolist()) # 각 막대들의 높이
    else :
        locAll_axis_x = [*loc_latest, *loc_previous ] #막대들이 위치할 x 좌표
        locAll_axis_y = joinLists(df_column3_row2.values.tolist()) # 각 막대들의 높이
    # 컬러#0042ED
    color_latest = '#000093'
    color_previous = '#0054FF'
    color_pprevious = '#6CC0FF'#000093
    #color_previous = '#38C6F4'
    
    # 막대 그리기
    if len(df_column3_row2.index) ==3:
        plt.bar(loc_latest, df_column3_row2.iloc[0,], color = color_latest, width = barWidth, edgecolor = color_latest, 
                label = '%s'%df_column3_row2.index[0])

        plt.bar(loc_previous, df_column3_row2.iloc[1,], color = color_previous, width = barWidth, edgecolor = color_previous, 
                label = '%s'%df_column3_row2.index[1])

        plt.bar(loc_pprevious, df_column3_row2.iloc[2,], color = color_pprevious, width = barWidth, edgecolor = color_pprevious, 
                label = '%s'%df_column3_row2.index[2])
    elif len(df_column3_row2.index) ==2:
        plt.bar(loc_latest, df_column3_row2.iloc[0,], color = color_latest, width = barWidth, edgecolor = color_latest, 
                label = '%s'%df_column3_row2.index[0])

        plt.bar(loc_previous, df_column3_row2.iloc[1,], color = color_previous, width = barWidth, edgecolor = color_previous, 
                label = '%s'%df_column3_row2.index[1])

    
    
    # 각 막대 위에 값 표시와 표시 위치 세부 조정
    for index in range(0, len(locAll_axis_x)):
        if money_scaler(locAll_axis_y[index])[:2] =='손실':
            if locAll_axis_y[index] > 0 :
                    ax.text(x = locAll_axis_x[index], y = locAll_axis_y[index] + (max(locAll_axis_y)*0.03), 
                            s = '-' + money_scaler(locAll_axis_y[index])[3:], horizontalalignment='center', size = 10)
            else :
                    ax.text(x = locAll_axis_x[index], y = locAll_axis_y[index] + (locAll_axis_y[index]*(-1.5)), 
                            s = '-' + money_scaler(locAll_axis_y[index])[3:], horizontalalignment='center', size = 10)
        else:
            if locAll_axis_y[index] > 0 :
                    ax.text(x = locAll_axis_x[index], y = locAll_axis_y[index] + (max(locAll_axis_y)*0.03), 
                            s =  money_scaler(locAll_axis_y[index])[3:], horizontalalignment='center', size = 10)
            else :
                    ax.text(x = locAll_axis_x[index], y = locAll_axis_y[index] + (locAll_axis_y[index]*(-1.5)), 
                            s =  money_scaler(locAll_axis_y[index])[3:], horizontalalignment='center', size = 10)
                
    #plt.ylabel('단위: ' + unit, rotation = 'horizontal', size = 12) # ---- 이 두 줄은 y축에 이름표 달고 위치 조정하기
    ax.yaxis.set_label_coords(-0.03, 1.05) 

    # x, y축 조정
    if min(locAll_axis_y) < 0 and max(locAll_axis_y) > 0 :
        plt.ylim((min(locAll_axis_y) * 1.25, max(locAll_axis_y) * 2))
    elif min(locAll_axis_y) > 0 and max(locAll_axis_y) > 0 :
        plt.ylim( (0, max(locAll_axis_y) * 1.25) )
    else :
        plt.ylim( (0, max(locAll_axis_y) * 1.25) )
    plt.yticks(fontsize = 11, color = '#4F4F4F')
    plt.xticks(fontsize = 15)
    ax.tick_params(axis='both', which='both', length=0)

    ax.yaxis.grid(True, linewidth = 0.5)
    ax.set_axisbelow(True)
    ## 대표님 요청사항 01/16 y축 숫자 제거, 단위제거 => Set visible
    ax.yaxis.set_visible(False)
    # X축 label을 표시하고, 그것의 위치를 가로, 세로로 미세 이동하는 코드
    if len(df_column3_row2)==2:
        plt.xticks([r + barWidth for r in loc_latest - 0.25], df_column3_row2.columns, fontweight = 'light')
    elif len(df_column3_row2)==3:
        plt.xticks([r + barWidth for r in loc_latest - 0.125], df_column3_row2.columns, fontweight = 'light')
    for i in range(0, len(ax.xaxis.get_majorticklabels())):
                   ax.xaxis.get_majorticklabels()[i].set_y(-.02)
    # y축 label의 포맷을 정해줌: 세 자리 단위로 쉼표로 끊김
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # y-tick들 가로로 미세 조정
    for i in range(0, len(ax.yaxis.get_majorticklabels())):
                   ax.yaxis.get_majorticklabels()[i].set_x(-.01)

    # y-tick들 듬성듬성하게 display : yticks 갯수가 너무 많으면(>6) 절반만 표시되게 하자
    if len(ax.yaxis.get_majorticklabels()) > 6:
        ax.locator_params(nbins = round(len(ax.yaxis.get_majorticklabels()) / 2) , axis='y')
    else:
        ax.locator_params(nbins = len(ax.yaxis.get_majorticklabels()), axis = 'y')
    
    # 불필요한 까만 테두리 없애기
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 모든 값이 양수이면 (최소값이 0 이상) 아래 테두리만 남겨주기
    if min(locAll_axis_y) >= 0:
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('gray')
    else: 
        ax.spines['bottom'].set_visible(False)
    
    # 범주 박스 표시
    plt.legend()
    ax.legend(loc='upper center', bbox_to_anchor = (0.5, -0.1),
          fancybox = True, shadow = True, ncol = 5, fontsize = 12,
         handlelength = 1, handleheight = 1)

    # 끝

    image = plt.gcf()
    plt.show()
    image.savefig('chart/' + rcp_no + '_' + no + '.png', bbox_inches="tight")
    Image.open('chart/' + rcp_no + '_' + no + '.png').convert("RGB").save('chart/' + rcp_no + '_' + no + '.jpg')
    os.remove('chart/' + rcp_no + '_' + no + '.png')



def annual_earning_image(sonic_df, rcp_no,crp_nm,rpt_nm):
    fig1 = plt.figure(figsize=(10,6))
    ax = fig1.add_subplot(111)


    earning_han = ['매출액', '영업이익', '당기순이익']
    earning_eng = ['sales', 'operating_income', 'net_income']
    my_df = pd.DataFrame()
    for earning in range(0,3):
        earn_han = earning_han[earning]
        earn_eng = earning_eng[earning]

        my_df_dict = {'                '+ earning_han[earning] : list(sonic_df.loc[earn_han])}
        my_df_ = pd.DataFrame(OrderedDict(sorted(my_df_dict.items())))
        this_year = re.sub('[^0-9]','',rpt_nm)[:4] +'년'
        last_year = str(int(re.sub('[^0-9]','',rpt_nm)[:4])-1) +'년'
        llast_year = str(int(re.sub('[^0-9]','',rpt_nm)[:4])-2) +'년'
        if len(sonic_df.columns)==3:
            my_df_ = my_df_.rename(index = {0:this_year  , 1:last_year,2: llast_year})
            my_df = pd.concat([my_df,my_df_],axis=1)
            years = '3'
        elif len(sonic_df.columns)==2:
            my_df_ = my_df_.rename(index = {0:this_year  , 1:last_year})
            my_df = pd.concat([my_df,my_df_],axis=1)
            years = '2'
    title_text = str(crp_nm +' 최근 '+ years + '개년 실적추이')
    no = '1'
    plot_bar_sales(my_df, title_text, rcp_no, no)
def semiannual_earning_image(sonic_df, sonic_shape, rcp_no,crp_nm):
    
    earning_han = ['매출액', '영업이익', '당기순이익']
    earning_eng = ['sales', 'operating_income', 'net_income']
    my_df = pd.DataFrame()
    this_year = re.sub('[^0-9]','',rpt_nm)[:4] +'년'
    last_year = str(int(re.sub('[^0-9]','',rpt_nm)[:4])-1) +'년'
    llast_year = str(int(re.sub('[^0-9]','',rpt_nm)[:4])-2) +'년'
    period_text = {'annual':this_year, 'semiannual':'반기(누적)', 'quarter': sonic.iloc[0,1][-3:] + '(누적)'}
    period_account = {'annual':'당기', 'semiannual':'누적', 'quarter':'누적'}
    for earning in range(0,3):
        earn_han = earning_han[earning]
        earn_eng = earning_eng[earning]
        my_df_dict = {'                '+ earning_han[earning] : list(sonic.loc[earn_han])}
        my_df_ = pd.DataFrame(OrderedDict(sorted(my_df_dict.items())))
        if sonic_shape==3:
            my_df = pd.concat([my_df,my_df_],axis=1)
            my_df = my_df.reset_index(drop=True)
            my_df = my_df.iloc[[1,3,5],:]
            #my_df.index =[this_year +' '+ period_text[period],last_year +' '+period_text[period],llast_year +' '+period_text[period]]
            years = '3'
        elif sonic_shape==2:  
            my_df = pd.concat([my_df,my_df_],axis=1)
            my_df = my_df.reset_index(drop=True)
            my_df = my_df.iloc[[1,3],:]
            #my_df=my_df.reindex([this_year +' '+ period_text[period],last_year +' '+period_text[period]] )
            #my_df.index =[this_year +' '+ period_text[period],last_year +' '+period_text[period]]       
            years = '2'
    if sonic_shape==2:
        my_df.index =[this_year +' '+ period_text[period],last_year +' '+period_text[period]]
    elif sonic_shape == 3:
        my_df.index =[this_year +' '+ period_text[period],last_year +' '+period_text[period],llast_year +' '+period_text[period]]
    title_text = crp_nm +' 최근 '+years +'개년 실적추이'
    no = '1'
    plot_bar_sales(my_df, title_text, rcp_no, no)
def debt_ratio_image(jaemu_df, period, rcp_no, rpt_nm):
    fig1 = plt.figure(figsize=(10,6))
    ax = fig1.add_subplot(111)
    this_year = re.sub('[^0-9]','',rpt_nm)[:4] +'년'
    last_year = str(int(re.sub('[^0-9]','',rpt_nm)[:4])-1) +'년'
    llast_year = str(int(re.sub('[^0-9]','',rpt_nm)[:4])-2) +'년'
    period_text = {'annual':this_year, 'semiannual':'반기(누적)', 'quarter': sonic.iloc[0,1][-3:] + '(누적)'}
    period_account = {'annual':'당기', 'semiannual':'누적', 'quarter':'누적'}
    
    earning_han = ['자본총계', '부채총계', '자산총계']
    my_df = pd.DataFrame()
    for earning in range(0,3):
        earn_han = earning_han[earning]

        my_df_dict = {'                '+ earning_han[earning] : list(jaemu_df.loc[earn_han])}
        my_df_ = pd.DataFrame(OrderedDict(sorted(my_df_dict.items())))
        this_year = re.sub('[^0-9]','',rpt_nm)[:4] +'년'
        last_year = str(int(re.sub('[^0-9]','',rpt_nm)[:4])-1) +'년'
        llast_year = str(int(re.sub('[^0-9]','',rpt_nm)[:4])-2) +'년'
        my_df_ = my_df_.rename(index = {0:this_year +' '+ period_text[period]  , 1:last_year +' '+ period_text[period],2: llast_year +' '+ period_text[period]})
        my_df = pd.concat([my_df,my_df_],axis=1)
    title_text = '자산 변동 추이'
    no = '2'
    plot_bar_sales(my_df, title_text, rcp_no, no)
def debt_ratio_comp_image(jaemu_df, period, rcp_no,rpt_nm):
    fig, ax = plt.subplots(figsize=(10,6))
    period_account = {'annual':'당기', 'semiannual':'당반기', 'quarter':'당분기'}
    plt.clf()
    plt.cla()
    plt.close()
    plotdata = list((jaemu_df.loc['부채총계'] / jaemu_df.loc['자본총계'])*100)
    plotindex = np.arange(len(plotdata))
    plt.plot(plotdata, marker = 'o')
    plt.ylim(0, math.ceil(max(plotdata)+0.5*(max(plotdata)-min(plotdata))))
    if period == 'annual':
        this_year = re.sub('[^0-9]','',rpt_nm)[:4] +'년'
        last_year = str(int(re.sub('[^0-9]','',rpt_nm)[:4])-1) +'년'
        llast_year = str(int(re.sub('[^0-9]','',rpt_nm)[:4])-2) +'년'
        if len(jaemu_df.columns) == 2:
            plt.xlim(-0.5, 1.5)
            plt.bar(plotindex, (0,0), tick_label=(this_year, last_year), align='center')
        if len(jaemu_df.columns) == 3: 
            plt.xlim(-0.5, 2.5)
            plt.bar(plotindex, (0,0,0), tick_label=(this_year,last_year, llast_year), align='center')
        plt.title('부채비율(부채/자본) 변동 추이',size = 16)
        for a,b in zip(plotindex, plotdata): 
            plt.text(a, b, str(round(b))+'%',ha='center', va='bottom',size = 12)
        image = plt.gcf()
        plt.ylabel('단위: %',rotation = 'horizontal' , size = 12,horizontalalignment='left', position=(1,1.03))
        plt.show()
        image.savefig('chart/' + rcp_no + '_3' + '.png', bbox_inches="tight")
        Image.open('chart/' + rcp_no + '_3' + '.png').convert("RGB").save('chart/' + rcp_no + '_3' + '.jpg')
        os.remove('chart/' + rcp_no + '_3' + '.png')
    else :    
        if len(jaemu_df.columns) == 2:
            plt.xlim(-0.5, 1.5)
            plt.bar(plotindex, (0,0), tick_label=('전기', period_account[period]), align='center')
        if len(jaemu_df.columns) == 3: 
            plt.xlim(-0.5, 2.5)
            plt.bar(plotindex, (0,0,0), tick_label=('전전기','전기', period_account[period]), align='center')
        plt.title('부채비율(부채/자본) 변동 추이')
        image = plt.gcf()
        plt.show()
        image.savefig('chart/' + rcp_no + '_3' + '.png', bbox_inches="tight")
        Image.open('chart/' + rcp_no + '_3' + '.png').convert("RGB").save('chart/' + rcp_no + '_3' + '.jpg')
        os.remove('chart/' + rcp_no + '_3' + '.png')


# In[195]:


def financial_statements_soup(rcp_no, name = '재무제표'):
    url = "http://dart.fss.or.kr/dsaf001/main.do?rcpNo="+str(rcp_no)
    req = requests.get(url)
    html = req.text

    comp = re.compile('new Tree\.TreeNode\({\r\n\t\t\ttext: ".*"')
    tree = comp.findall(html)
    ind = str()
    for j in range(0,len(tree)):
        ind = ind +  "\n" +  tree[j][31:-1] + "\n"

    comp = re.compile('\n.*\n')
    index = comp.findall(ind)
    for j in range(0,len(index)):
        if index[j].startswith(name + '\n',4)== True:
            page= j

    comp = re.compile('viewDoc\(.*\)')
    viewdoc = comp.findall(html)
    vdurl = removeword(viewdoc)

    if (len(vdurl) == 4):
        vdurl_list = makeurl(vdurl, 2)
    elif (vdurl[1] == vdurl[-3]):
        vdurl_list = makeurl(vdurl, 3)
    else:
        vdurl_list = makeurl(vdurl, 2)

    inreq = requests.get(vdurl_list[page])
    inhtml = inreq.text
    soup = BeautifulSoup(inhtml, 'lxml')
    return soup


# In[16]:


def removeword(viewdoc):
    vdurl = []
    for i in range(0,len(viewdoc)):
        remove0_code = viewdoc[i][8:-1].replace("'",'')
        remove1_code = remove0_code.replace(' ','')
        split_code = remove1_code.split(',')
        vdurl.append(split_code)
    return vdurl
def makeurl(vdurl, cut):
    vdurl_list = []
    for i in range(1, len(vdurl)-cut):
        s0 = "rcpNo=%s&dcmNo=%s&" % (vdurl[i][0], vdurl[i][1])
        s1 = "eleId=%s&offset=%s&" % (vdurl[i][2], vdurl[i][3])
        s2 = "length=%s&dtd=%s" % (vdurl[i][4], vdurl[i][5])
        url = "http://dart.fss.or.kr/report/viewer.do?" +s0+s1+s2
        vdurl_list.append(url)
    return vdurl_list


# In[14]:



import pandas as pd
import dart_api
import re
import requests
from bs4 import BeautifulSoup
from html_table_parser import parser_functions as parser
from collections import Counter
def financial_statements_soup(rcp_no, name = '재무제표'):
    url = "http://dart.fss.or.kr/dsaf001/main.do?rcpNo="+str(rcp_no)
    req = requests.get(url)
    html = req.text

    comp = re.compile('new Tree\.TreeNode\({\r\n\t\t\ttext: ".*"')
    tree = comp.findall(html)
    ind = str()
    for j in range(0,len(tree)):
        ind = ind +  "\n" +  tree[j][31:-1] + "\n"

    comp = re.compile('\n.*\n')
    index = comp.findall(ind)
    for j in range(0,len(index)):
        if index[j].startswith(name + '\n',4)== True:
            page= j

    comp = re.compile('viewDoc\(.*\)')
    viewdoc = comp.findall(html)
    vdurl = removeword(viewdoc)

    if (len(vdurl) == 4):
        vdurl_list = makeurl(vdurl, 2)
    elif (vdurl[1] == vdurl[-3]):
        vdurl_list = makeurl(vdurl, 3)
    else:
        vdurl_list = makeurl(vdurl, 2)

    inreq = requests.get(vdurl_list[page])
    inhtml = inreq.text
    soup = BeautifulSoup(inhtml, 'lxml')
    return soup


# In[407]:


def num_emp_count (emp,sex,origin):
    emp_male = emp[emp['성별']== sex][[origin]]
    if len(emp_male)==0:
        emp_male = pd.DataFrame({'g':[0]})
    else :
        emp_male = emp_male.apply(lambda x: [re.sub('년.*', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub('\..*', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub('-', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub(',', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub('.*월', '', xx) for xx in x])
        emp_male = emp_male.dropna()
        if len(emp_male)!=0:
            emp_male[emp_male[origin].str.contains('개월') ==False]
            emp_male = emp_male.apply(lambda x: x.replace('', 0))
            emp_male = emp_male[emp_male[origin]!='']
            emp_male = emp_male.astype('int64')
            emp_male = emp_male[emp_male[origin]!=0]
            emp_male = emp_male.reset_index(drop=True)
            if emp_male.loc[0][0] > 1000:
                emp_male = emp_male.astype('str')
                emp_male = emp_male.apply(lambda x: [int(datetime.datetime.now().strftime('%Y'))-int(xx[:4]) for xx in x])
    return len(emp_male)
def num_emp (emp,sex,origin):
    emp_male = emp[emp['성별']== sex][[origin]]
    if len(emp_male)==0:
        emp_male = pd.DataFrame({'g':[0]})
    else :
        emp_male = emp_male.apply(lambda x: [re.sub('년.*', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub('\..*', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub('-', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub(',', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub('.*월', '', xx) for xx in x])
        emp_male = emp_male.dropna()
        if len(emp_male)!=0:
            emp_male[emp_male[origin].str.contains('개월') ==False]
            emp_male = emp_male.apply(lambda x: x.replace('', 0))
            emp_male = emp_male[emp_male[origin]!='']
            emp_male = emp_male.astype('int64')
            emp_male = emp_male[emp_male[origin]!=0]
            emp_male = emp_male.reset_index(drop=True)
            if emp_male.loc[0][0] > 1000:
                emp_male = emp_male.astype('str')
                emp_male = emp_male.apply(lambda x: [int(datetime.datetime.now().strftime('%Y'))-int(xx[:4]) for xx in x])
    return list(emp_male.sum())[0]
def num_emp_co (emp,sex,origin):
    emp_male = emp[emp['성별']== sex][[origin]]
    if len(emp_male)==0:
        emp_male = pd.DataFrame({'g':[0]})
    else :
        emp_male = emp_male.apply(lambda x: [re.sub('년.*', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub('\..*', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub('-', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub(',', '', xx) for xx in x])
        emp_male = emp_male.apply(lambda x: [re.sub('.*월', '', xx) for xx in x])
        emp_male = emp_male.dropna()
        if emp_male.columns[0]==emp_male.columns[1]:
            emp_male = pd.DataFrame(emp_male.iloc[:,0])
        if len(emp_male)!=0:
            emp_male[emp_male[origin].str.contains('개월') ==False]
            emp_male = emp_male.apply(lambda x: x.replace('', 0))
            #emp_male = emp_male[emp_male[origin]!='']
            emp_male = emp_male.astype('int64')
            emp_male = emp_male[emp_male[origin]!=0]
            emp_male = emp_male.reset_index(drop=True)
    return list(emp_male.sum())[0]
def category_age(x):
    if x < 10:
        return '오류'
    elif x < 20:
        return '10대'
    elif x < 30:
        return '20대'
    elif x < 40:
        return '30대'
    elif x < 50:
        return '40대'
    elif x < 60:
        return '50대'
    elif x < 70:
        return '60대'
    else:
        return '70대 이상'
def top_emp (emp,origin):
    emp_male = emp[[origin]]
    emp_male = emp_male.apply(lambda x: [re.sub('[^0-9]', '', xx) for xx in x])
    emp_male = emp_male.apply(lambda x: x.replace('', 0))
    emp_male = emp_male.iloc[2:]
    emp_male = emp_male.reset_index(drop=True)
    emp_male = emp_male.apply(lambda x: [ 2019 - int(xx[:4]) for xx in x])
    emp_male['출생년월'] = emp_male['출생년월'].apply(category_age)
    emp_male = emp_male.sort_values(by='출생년월')
    emp_male = emp_male.reset_index(drop=True)
    return emp_male


# In[189]:


def financial_statements_data(soup, name):
    findis = re.compile(name)
    isstyle = re.compile('.*font-weight:.*;')
    wonstyle = re.compile('.*단위.*:.*원.*')
    try:
        findsoup = soup.find('p', attrs = {'style':isstyle}, text = findis).find_all_next()
    except:
        try:
            findsoup = soup.find('span', attrs = {'style':isstyle}, text = findis).find_all_next()
        except:
            try:
                findsoup = soup.find('td', text = findis).find_all_next()
            except:
                findsoup = soup.find('p', text = findis).find_all_next()
    soupis = BeautifulSoup(str(findsoup),'lxml')
    wonstyle = re.compile('.*단위.*:.*원.*')
    findwon = soupis.find(text = wonstyle)
    unit = re.sub('[^가-힣]', '', findwon)
    unit = re.sub('단위','',unit)

    temp = soupis.find('table',attrs={'border':'1'})
    p=parser.make2d(temp)
    fs_data = pd.DataFrame(p,columns=p[0])
    
    return fs_data, unit


# In[473]:


def sex_ratio(emp,num):
    data = [abs(num_emp(emp, '남','합 계')),abs(num_emp(emp, '여','합 계'))]
    plt.pie(data,labels = ['남','여'],  startangle=90,colors =['#1F50B5','#FF5200']) #colors = colors_latest, #width = barWidth, #edgecolors = colors_latest, #label = '%s'%df_column3_row2.index[0])
    plt.title('                                    '+crp_nm +' 직원성별비율                      (단위 : 퍼센트)')
    plt.axis('equal')
    image = plt.gcf()
    plt.close()
    no=num
    image.savefig('chart/' + rcp_no + '__' + no + '.png', bbox_inches="tight")
    Image.open('chart/' + rcp_no + '__' + no + '.png').convert("RGB").save('chart/' + rcp_no + '__' + no + '.jpg')
    os.remove('chart/' + rcp_no + '__' + no + '.png')
def earning(emp,num):
    data = [abs(num_emp(emp, '남','1인평균급여액'))/num_emp_count(emp, '남','1인평균급여액'),abs(num_emp(emp, '여','1인평균급여액'))/num_emp_count(emp, '여','1인평균급여액'),
            abs(num_emp(emp, '합 계','1인평균급여액'))/num_emp_count(emp, '합 계','1인평균급여액')]
    index = ['남','여','전체 평균']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(index, data,label='직원현황',color =['#1F50B5','#FF5200','grey'])
    plt.ylabel('(단위 : '+ob_unit+')', rotation = 0)
    ax.yaxis.set_label_coords(-0.09, 1.02)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(crp_nm + ' 1인 평균급여액')#colors = colors_latest, #width = barWidth, #edgecolors = colors_latest, #label = '%s'%df_column3_row2.index[0])
    image = plt.gcf()
    plt.close()
    no=num
    image.savefig('chart/' + rcp_no + '__' + no + '.png', bbox_inches="tight")
    Image.open('chart/' + rcp_no + '__' + no + '.png').convert("RGB").save('chart/' + rcp_no + '__' + no + '.jpg')
    os.remove('chart/' + rcp_no + '__' + no + '.png')
def years(emp,num):
    data = [abs(num_emp(emp, '남','평 균근속연수'))/num_emp_count(emp, '남','평 균근속연수'),abs(num_emp(emp, '여','평 균근속연수'))/num_emp_count(emp, '여','평 균근속연수'),
            abs(num_emp(emp, '합 계','평 균근속연수'))/num_emp_count(emp, '합 계','평 균근속연수')]
    index = ['남','여','전체 평균']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(index, data,label='직원현황',color =['#1F50B5','#FF5200','grey'])
    plt.ylabel('(단위 : 년)', rotation = 0)
    ax.yaxis.set_label_coords(-0.09, 1.02)
    plt.title(crp_nm + ' 평균 근속연수')#colors = colors_latest, #width = barWidth, #edgecolors = colors_latest, #label = '%s'%df_column3_row2.index[0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    image = plt.gcf()
    plt.close()
    no=num
    image.savefig('chart/' + rcp_no + '__' + no + '.png', bbox_inches="tight")
    Image.open('chart/' + rcp_no + '__' + no + '.png').convert("RGB").save('chart/' + rcp_no + '__' + no + '.jpg')
    os.remove('chart/' + rcp_no + '__' + no + '.png')
def count_emp(emp,num):
    data = [abs(num_emp_co(emp, '남','기간의정함이 없는근로자')),abs(num_emp_co(emp, '여','기간의정함이 없는근로자')),
            abs(num_emp_co(emp, '남','기간제근로자')),abs(num_emp_co(emp, '여','기간제근로자'))]
    index = ['정규직(남)','정규직(여)','기간제(남)','기간제(여)']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(index, data,color=['#1F50B5','#FF5200','#1F50B5','#FF5200'])
    plt.ylabel('(단위 : 명)', rotation = 0)
    ax.yaxis.set_label_coords(-0.09, 1.02)
    plt.title(crp_nm + ' 정규직/비정규직 직원 현황')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    image = plt.gcf()
    plt.close()
    no=num
    image.savefig('chart/' + rcp_no + '__' + no + '.png', bbox_inches="tight")
    Image.open('chart/' + rcp_no + '__' + no + '.png').convert("RGB").save('chart/' + rcp_no + '__' + no + '.jpg')
    os.remove('chart/' + rcp_no + '__' + no + '.png')
def ob_ages(emp,num):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.countplot(x='출생년월', data=top_emp(emp,'출생년월'))
    plt.title(crp_nm +' 임원 연령 분포')
    plt.ylabel('(단위 : 명)', rotation = 0)
    ax.yaxis.set_label_coords(-0.09, 1.02)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('')
    image = plt.gcf()
    plt.close()
    no=num
    image.savefig('chart/' + rcp_no + '__' + no + '.png', bbox_inches="tight")
    Image.open('chart/' + rcp_no + '__' + no + '.png').convert("RGB").save('chart/' + rcp_no + '__' + no + '.jpg')
    os.remove('chart/' + rcp_no + '__' + no + '.png')


# # 직원현황 메일보내기

# In[471]:



for i in range(0,len(report_df)):
    crp_cd, crp_cls, crp_nm, rcp_dt, rcp_dt, rcp_no, rpt_nm, period = make_object(report_df,i)
    soup = fs.financial_statements_soup(rcp_no, name = '임원 및 직원의 현황')
    ob_, ob_unit = financial_statements_data(soup, name = '.*임.*원.*현.*황.*')
    for k in range(0,len(soup.find_all('table'))):
        find = str(soup.find_all('table')[k])
        if re.search('의결권',find):
            ob_soup = soup.find_all('table')[k]
            ob_soup = BeautifulSoup(str(ob_soup),'lxml')
        elif re.search('직 원 수',find):
            yb_soup = soup.find_all('table')[k]
            yb_soup = BeautifulSoup(str(yb_soup),'lxml')

    p=parser.make2d(yb_soup)
    emp = pd.DataFrame(p,columns=p[0])
    emp.columns = emp.loc[1]
    sex_ratio(emp,'1')
    earning(emp,'2')
    years(emp,'3')
    count_emp(emp,'4')
    p=parser.make2d(ob_soup)
    emp = pd.DataFrame(p,columns=p[0])
    emp.columns = emp.loc[1]
    ob_ages(emp,'5')
    msg =MIMEMultipart('related')
    msg['Subject'] = Header('[test]'+crp_nm+' 사업보고서 임직원 현황 시각화 테스트중', 'utf-8')
    msg['From'] = formataddr((str(Header(u'M.Robo', 'utf-8')), 'm.robo.walt@gmail.com'))
    msg['To'] = ','.join(reciplist)
    msg_alternative = MIMEMultipart('alternative')
    msg.attach(msg_alternative)

    #첨부파일
    #with open('./xml/' + rcp_no + '.xml', 'rb') as opened:
    #    openedfile = opened.read()
    #attachedfile = MIMEApplication(openedfile, _subtype = "pdf")
    #attachedfile.add_header('content-disposition', 'attachment', filename = rcp_no + '.xml')
    #msg.attach(attachedfile)
    gongsi_href = '<a href='+'"http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '">공시 전문으로 이동</a>'
    mail_text = crp_nm+' 사업보고서 임직원 현황 시각화 테스트중'+'<br>'+gongsi_href
    #이미지첨부
    for no in range(1,6):
        img = dict(path = 'chart/' + rcp_no + '__' + str(no) + '.jpg', cid = str(uuid.uuid4()))
        mail_text += u'<div dir="ltr">''<img src="cid:{cid}" alt="{alt}"></div>'.format(alt=cgi.escape(img['path'], quote=True), **img)
        try:
            with open(img['path'], 'rb') as file:
                msg_image = MIMEImage(file.read(), name = os.path.basename(img['path']))
                msg_image.add_header('Content-ID', '<{}>'.format(img['cid']))
            msg.attach(msg_image)
        except:
            pass

    mail_text = MIMEText(mail_text, 'html', 'utf-8')
    msg_alternative.attach(mail_text)


    mailServer = smtplib.SMTP('smtp.gmail.com', 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.ehlo()
    mailServer.login("m.robo.walt" , "***")
    mailServer.sendmail('m.robo.walt@gmail.com', reciplist, msg.as_string())
    mailServer.quit()
    print(str(i)+' Mailing complite')

