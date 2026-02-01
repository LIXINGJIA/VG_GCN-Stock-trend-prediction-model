import tushare as ts
import pandas as pd
import os
token="your token" #tushare token
pro = ts.pro_api(token)



big_100_stock_codes = [
    "600016.SH", "600088.SH", "600078.SH", "600026.SH", "600006.SH",
    "600053.SH", "600081.SH", "600062.SH", "600094.SH", "600022.SH",
    "600054.SH", "600064.SH", "600066.SH", "600101.SH", "600123.SH",
    "600130.SH", "600082.SH", "600096.SH", "600125.SH", "600012.SH",
    "600008.SH", "600015.SH", "600019.SH", "600020.SH", "600028.SH",
    "600029.SH", "600107.SH", "600057.SH", "600075.SH", "600103.SH",
    "600120.SH", "600129.SH", "600018.SH", "600033.SH", "600076.SH",
    "600126.SH", "600072.SH", "600009.SH", "600021.SH", "600099.SH",
    "600104.SH", "600128.SH", "600051.SH", "600059.SH", "600089.SH",
    "600100.SH", "600113.SH", "600000.SH", "600017.SH", "600039.SH",
    "000012.SZ", "000586.SZ", "000020.SZ", "000544.SZ", "000510.SZ",
    "000029.SZ", "000554.SZ", "000558.SZ", "000036.SZ", "000430.SZ",
    "000507.SZ", "000521.SZ", "000537.SZ", "000560.SZ", "000638.SZ",
    "000561.SZ", "000089.SZ", "000541.SZ", "000652.SZ", "000048.SZ",
    "000419.SZ", "000559.SZ", "000570.SZ", "000576.SZ", "000593.SZ",
    "000552.SZ", "000630.SZ", "000009.SZ", "000011.SZ", "000019.SZ",
    "000050.SZ", "000100.SZ", "000422.SZ", "000523.SZ", "000545.SZ",
    "000547.SZ", "000008.SZ", "000021.SZ", "000031.SZ", "000069.SZ",
    "000156.SZ", "000403.SZ", "000526.SZ", "000573.SZ", "000655.SZ",
    "000506.SZ", "000039.SZ", "000058.SZ", "000059.SZ", "000090.SZ",
    "600563.SH", "601998.SH", "000919.SZ", "600269.SH", "002096.SZ",
    "000027.SZ"
]

stocks_list=['000012.SZ', '000586.SZ', '000020.SZ' ,'000544.SZ', '000510.SZ', '000029.SZ',
 '000554.SZ' ,'000558.SZ' ,'000036.SZ' ,'000430.SZ' ,'000507.SZ' ,'000521.SZ',
 '000537.SZ' ,'000560.SZ' ,'000638.SZ' ,'000561.SZ' ,'000089.SZ' ,'000541.SZ',
 '000652.SZ' ,'000048.SZ' ,'000419.SZ' ,'000559.SZ' ,'000570.SZ' ,'000576.SZ',
 '000593.SZ' ,'000552.SZ' ,'000630.SZ' ,'000009.SZ' ,'000011.SZ' ,'000019.SZ',
 '000050.SZ' ,'000100.SZ' ,'000422.SZ' ,'000523.SZ' ,'000545.SZ' ,'000547.SZ',
 '000008.SZ' ,'000021.SZ' ,'000031.SZ' ,'000069.SZ' ,'000156.SZ' ,'000403.SZ',
 '000526.SZ' ,'000573.SZ' ,'000655.SZ' ,'000506.SZ' ,'000039.SZ' ,'000058.SZ',
 '000059.SZ' ,'000090.SZ']
#需要添加大规模股票的股票
stocks_list_2=[
"600563.SH",
"601998.SH",
"000919.SZ",
"600269.SH",
"002096.SZ",
"000027.SZ",
]
stock_list_test=['600016.SH',
'600088.SH',
'600078.SH',
'600026.SH',
'600006.SH',
'600053.SH',
'600081.SH',
]


# 600563.SH	57.89%	52.14%	48.80%	50.41%
# 601998.SH	57.54%	53.57%	35.43%	42.65%
# 000919.SZ	56.84%	61.95%	46.67%	53.24%
# …	…	…	…	…
# 600269.SH	49.82%	42.59%	36.22%	39.15%
# 002096.SZ	49.65%	51.02%	34.72%	41.32%
# 000027.SZ	49.47%	50.72%	47.95%	49.30%
#大规模股票中包含沪深300的股票

stock_list_3=[
    '600016.SH',
    '600015.SH',
    '600019.SH',
    '600028.SH',
    '600029.SH',
    '600018.SH',
    '600009.SH',
    '600104.SH',
    '600089.SH',
    '600000.SH',
    '600039.SH',
    '000100.SZ',
    '000069.SZ',
    '601998.SH',
]

#沪深300权重
#       con_code  weight
# 29   600028.SH  0.6830
# 34   600016.SH  0.5972
# 52   600000.SH  0.4900
# 55   000100.SZ  0.4817
# 63   600089.SH  0.4341
# 70   600104.SH  0.3932
# 90   600019.SH  0.3323
# 113  600009.SH  0.2546
# 124  600015.SH  0.2356
# 155  600029.SH  0.1869
# 232  601998.SH  0.1160
# 259  600018.SH  0.0945
# 266  000069.SZ  0.0879
# 269  600039.SH  0.0853
#14个
#中证500权重
# 600062.SH 0.131
# 600022.SH 0.126
# 600066.SH 0.364
# 600096.SH 0.366
# 600008.SH 0.217
# 600129.SH 0.237
# 600126.SH 0.11
# 600021.SH 0.202
# 000537.SZ 0.092
# 000089.SZ 0.115
# 000559.SZ 0.104
# 000630.SZ 0.415
# 000009.SZ 0.403
# 000050.SZ 0.185
# 000547.SZ 0.147
# 000021.SZ 0.201
# 000031.SZ 0.064
# 000156.SZ 0.112
# 000039.SZ 0.266
# 600563.SH 0.21
# 000027.SZ 0.16
#21个

        # "start_date": 20220101,
        # "end_date": 20240201,

def get_data(stock,path):

    path=os.path.join(path,stock+'.csv')
    df = pro.daily(**{
        "ts_code":stock ,
        "trade_date": "",
        "start_date":20190101,
        "end_date":20221231,
        "limit": "",
        "offset": ""
    }, fields=[
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "pct_chg"
    ])
    # df.insert(df.shape[1],"updown",0)
    # for i in range(df.shape[0]):
    #         if df.loc[i,"pct_chg"]>=0:
    #             df.loc[i,"updown"]=1


    # df=df.rename(columns={"trade_date":"date"})
    # df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    # df=df.drop("pct_chg",axis=1)
    # df["updown"]=df["updown"].shift(-1)
    # df=df.iloc[:-1]
    # df.to_csv(path,index=False)



    df=df.sort_values("trade_date")
    df.insert(df.shape[1],"updown",0)
    for i in range(df.shape[0]-1):
        today_close = df.iloc[i]["close"]  # 今天的收盘价
        tomorrow_close = df.iloc[i+1]["close"]  # 明天的收盘价
        if today_close <= tomorrow_close:
            # 今天的标签设为1（涨或平）
            df.iloc[i, df.columns.get_loc("updown")] = 1
    df = df.iloc[:-1]
    df=df.rename(columns={"trade_date":"date"})
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    print(df.head())
    df.to_csv(path,index=False)




def get_stock_codes(num=500,where="00"):
    """获取指定数量的A股股票代码（上交所+深交所）"""
    # 调用stock_basic接口获取股票列表
    # is_hs: 是否沪深港通标的（N否 H沪股通 S深股通），这里选所有A股
    # list_status: 上市状态（L上市 D退市 P暂停上市），选L
    df = pro.stock_basic(
        exchange='',
        list_status='L',  # 只取上市股票
        fields='ts_code, name'
    )
    
    # 筛选A股（排除B股、北交所等，A股代码以60、00、30开头）
    a_share_codes = []
    for code in df['ts_code']:
        if code.startswith(('60')):  # 上交所A股(60)、深交所A股(00/30)  去除sz只要sh'60', '00', 
            a_share_codes.append(code)
        if len(a_share_codes) >= num:
            break  # 取够指定数量即停止
    
    return a_share_codes
if __name__ == '__main__':
    DATA=["600809.SH",]

    # # code=get_stock_codes(num=100)
    # # print(code)
    for i in DATA:   
        root_path="data/"
        get_data(stock=i,path=root_path)


    # 沪深300股票组成获取
    # df = pro.index_weight(index_code='000905.SH',start_date='20240130',end_date='20240201')
    # print(df.head())
    # df.to_excel("data/index_weight.xlsx",index=False)
    


    # stock_1 = get_stock_codes(num=500)
    # stock_2 = get_stock_codes(num=500,where="60")
    # stock=stock_1+stock_2
    # for i in stock:
    #     for j in big_100_stock_codes:
    #         if i==j:
    #             stock=stock.remove(i)


    

    

