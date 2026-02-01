import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# --------------------------
# 1. 数据预处理模块（去掉tail(101)，避免数据截取导致无交易）
# --------------------------
def load_and_preprocess_data(file_path):
    """
    加载回测数据并进行预处理
    :param file_path: CSV文件路径（如'/mnt/000001.SZ.csv'）
    :return: 预处理后的DataFrame
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 核心列名映射（匹配你的CSV列名：日期、收盘价、预测值）
    col_mapping = {
        '日期': 'date',
        '收盘价': 'close_price',
        '预测值': 'signal'  # 1=上涨/持平（买入持有），0=下跌（卖出空仓）
    }
    # 保留必要列并改名
    df = df[list(col_mapping.keys())].rename(columns=col_mapping)
    
    # 日期格式处理
    df['date'] = pd.to_datetime(df['date'])
    # 排序（按日期升序）
    df = df.sort_values('date').reset_index(drop=True)
    # 检查缺失值
    if df.isnull().any().any():
        print(f"数据存在{df.isnull().sum().sum()}个缺失值，已自动删除")
        df = df.dropna()
    
    # 【关键修改】去掉tail(101)，避免截取后无信号切换
    df = df.tail(101).reset_index(drop=True)
    
    print(f"预处理后数据量：{len(df)} 行")
    print(f"信号分布：1（买入）={len(df[df['signal']==1])} 行，0（卖出）={len(df[df['signal']==0])} 行")
    return df

# --------------------------
# 2. 策略收益计算模块（修正持仓初始值和交易触发逻辑）
# --------------------------
def calculate_strategy_returns(df, transaction_cost=0.001, risk_free_rate=0.02):
    result_df = df.copy()
    # 计算每日收益率（收盘价环比）
    result_df['daily_return'] = result_df['close_price'].pct_change()
    
    # 基准策略逻辑不变
    result_df['benchmark_return'] = result_df['daily_return']
    result_df['benchmark_cum_return'] = (1 + result_df['benchmark_return']).cumprod() - 1
    
    # ---------------------- 修正交易触发逻辑 ----------------------
    # 1. 持仓逻辑：signal=1→持有（1），signal=0→空仓（0），信号次日生效
    # 初始持仓设为0（空仓），但保留第一行的position以便计算trade_flag
    result_df['position'] = result_df['signal'].shift(1).fillna(0)
    # 2. 交易触发：position从0→1（买入）或1→0（卖出），trade_flag=1
    result_df['trade_flag'] = result_df['position'].diff().abs()
    # 处理首行trade_flag（diff后首行为NaN，设为0）
    result_df['trade_flag'] = result_df['trade_flag'].fillna(0)
    # 3. 策略收益：持仓收益 - 交易成本（每次交易扣总资产的cost）
    result_df['strategy_return_raw'] = result_df['position'] * result_df['daily_return']
    result_df['strategy_return'] = result_df['strategy_return_raw'] - result_df['trade_flag'] * transaction_cost
    
    # 累计收益计算
    result_df['strategy_cum_return'] = (1 + result_df['strategy_return']).cumprod() - 1
    result_df['strategy_cum_return_raw'] = (1 + result_df['strategy_return_raw']).cumprod() - 1
    # 无风险收益（按日折算）
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    result_df['risk_free_return'] = daily_rf
    
    # 打印交易触发情况（调试用）
    trade_count = int(result_df['trade_flag'].sum())
    print(f"触发交易次数：{trade_count} 次")
    if trade_count > 0:
        trade_dates = result_df[result_df['trade_flag']==1]['date'].tolist()
        print(f"交易日期：{trade_dates[:5]}...")  # 打印前5次交易日期
    return result_df

# --------------------------
# 新增：年化波动率计算辅助函数
# --------------------------
def calculate_annual_volatility(daily_returns):
    """
    计算年化波动率（Ann. Vol.）
    :param daily_returns: 日收益率序列（Series）
    :return: 年化波动率（小数形式）
    """
    # 样本标准差（ddof=1，适配回测样本特性）
    daily_std = daily_returns.dropna().std(ddof=1)
    # 年化因子：√252（国际通用年交易日数）
    annual_vol = daily_std * np.sqrt(252)
    return annual_vol



# --------------------------
# 3. 核心指标计算模块（修正损益分析逻辑）
# --------------------------
def calculate_core_metrics(result_df,stock_code, risk_free_rate=0.02, transaction_cost=0.001):
    """
    计算夏普比率、累计回报率、损益分析等核心指标
    :param result_df: 含收益数据的DataFrame
    :param risk_free_rate: 无风险利率（年化）
    :return: 指标字典
    """
    # 筛选有效数据（保留所有行，仅去掉daily_return为NaN的行）
    valid_df = result_df.dropna(subset=['daily_return']).copy()
    trading_days = len(valid_df)
    annualization_factor = 252 / trading_days  # 年化因子
    
    # ---------------------- 1. 累计回报率指标 ----------------------
    cum_return_metrics = {
        'benchmark_total_return': valid_df['benchmark_cum_return'].iloc[-1],
        'strategy_total_return_raw': valid_df['strategy_cum_return_raw'].iloc[-1],
        'strategy_total_return': valid_df['strategy_cum_return'].iloc[-1],
        'benchmark_annual_return': (1 + valid_df['benchmark_cum_return'].iloc[-1]) ** (1/(trading_days/252)) - 1,
        'strategy_annual_return_raw': (1 + valid_df['strategy_cum_return_raw'].iloc[-1]) ** (1/(trading_days/252)) - 1,
        'strategy_annual_return': (1 + valid_df['strategy_cum_return'].iloc[-1]) ** (1/(trading_days/252)) - 1
    }
    
    # ---------------------- 2. 夏普比率 ----------------------
    valid_df['strategy_excess_return'] = valid_df['strategy_return'] - valid_df['risk_free_return']  # 扣成本的超额收益
    valid_df['strategy_excess_return_raw'] = valid_df['strategy_return_raw'] - valid_df['risk_free_return'] 
    # sharpe_metrics = {
    #     'benchmark_sharpe': (valid_df['benchmark_excess_return'].mean() / valid_df['benchmark_excess_return'].std()) * np.sqrt(annualization_factor) if valid_df['benchmark_excess_return'].std() !=0 else 0,
    #     'strategy_sharpe_raw': (valid_df['strategy_excess_return'].mean() / valid_df['strategy_excess_return'].std()) * np.sqrt(annualization_factor) if valid_df['strategy_excess_return'].std() !=0 else 0,
    #     'strategy_sharpe': (valid_df['strategy_excess_return'].mean() / valid_df['strategy_excess_return'].std()) * np.sqrt(annualization_factor) if valid_df['strategy_excess_return'].std() !=0 else 0
    #}
    # 修正后的代码
 # 未扣成本的超额收益

    sharpe_metrics = {
        'strategy_sharpe_raw': (valid_df['strategy_excess_return_raw'].mean() / valid_df['strategy_excess_return_raw'].std()) * np.sqrt(annualization_factor) if valid_df['strategy_excess_return_raw'].std() !=0 else 0,
        'strategy_sharpe': (valid_df['strategy_excess_return'].mean() / valid_df['strategy_excess_return'].std()) * np.sqrt(annualization_factor) if valid_df['strategy_excess_return'].std() !=0 else 0
    }
    # ---------------------- 3. 损益分析（修正核心逻辑） ----------------------
    # 提取所有交易记录（trade_flag=1的行）
    trade_records = valid_df[valid_df['trade_flag']==1].copy()
    total_trades = len(trade_records)
    trades = []  # 存储每笔交易的收益
    
    if total_trades > 0:
        # 关联交易对应的持仓方向（买入/卖出）：position=1→买入，0→卖出
        trade_records['trade_type'] = trade_records['position'].map({1: '买入', 0: '卖出'})
        # 计算每笔买入交易的收益（从买入到下一次卖出或回测结束）
        buy_dates = trade_records[trade_records['trade_type']=='买入']['date'].tolist()
        sell_dates = trade_records[trade_records['trade_type']=='卖出']['date'].tolist()
        
        # 处理买入-卖出对
        for i, buy_date in enumerate(buy_dates):
            # 找到该买入对应的卖出日期（下一次卖出）
            sell_date = next((d for d in sell_dates if d > buy_date), None)
            if sell_date:
                # 买入到卖出期间的收益
                buy_price = valid_df[valid_df['date']==buy_date]['close_price'].iloc[0]
                sell_price = valid_df[valid_df['date']==sell_date]['close_price'].iloc[0]
                trade_return = (sell_price / buy_price) - 1  # 持仓收益
                trade_return -= 2 * transaction_cost  # 扣除买入+卖出两次交易成本
                trades.append(trade_return)
            else:
                # 无对应卖出，计算到回测结束的收益（未平仓）
                buy_price = valid_df[valid_df['date']==buy_date]['close_price'].iloc[0]
                end_price = valid_df['close_price'].iloc[-1]
                trade_return = (end_price / buy_price) - 1  # 持仓收益
                trade_return -= transaction_cost  # 仅扣除买入成本
                trades.append(trade_return)
    
    # 损益分析指标
    winning_trades = len([r for r in trades if r > 0])
    losing_trades = len([r for r in trades if r < 0])
    profit_loss_metrics = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'average_trade_return': np.mean(trades) if total_trades > 0 else 0,
        'profit_factor': (np.sum([r for r in trades if r > 0]) / abs(np.sum([r for r in trades if r < 0]))) if losing_trades > 0 else 0
    }
    
    # ---------------------- 4. 最大回撤 ----------------------
    def calculate_max_drawdown(cum_returns):
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / (1 + rolling_max)
        return drawdown.min()
    
    drawdown_metrics = {
        'benchmark_max_drawdown': calculate_max_drawdown(valid_df['benchmark_cum_return']),
        'strategy_max_drawdown': calculate_max_drawdown(valid_df['strategy_cum_return']),
        'strategy_max_drawdown_raw': calculate_max_drawdown(valid_df['strategy_cum_return_raw'])
    }

        # ---------------------- 新增：5. 年化波动率（Ann. Vol.） ----------------------
    volatility_metrics = {
        # 基准策略年化波动率（基于基准日收益率）
        'benchmark_ann_vol': calculate_annual_volatility(valid_df['benchmark_return']),
        # 策略未扣成本年化波动率（基于未扣成本日收益率）
        'strategy_ann_vol_raw': calculate_annual_volatility(valid_df['strategy_return_raw']),
        # 策略扣成本年化波动率（基于扣成本日收益率）
        'strategy_ann_vol': calculate_annual_volatility(valid_df['strategy_return'])
    }

    
    # 合并所有指标
    all_metrics = {
        "stock_code":stock_code,
        **cum_return_metrics,
        **sharpe_metrics,
        **profit_loss_metrics,
        **drawdown_metrics,
        'trading_days': trading_days,
        **volatility_metrics
    }
    
    return all_metrics

# --------------------------
# 主函数
# --------------------------
def main(file_path,stock_code="lxj", transaction_cost=0.001, risk_free_rate=0.02):
    """
    主函数：执行完整回测分析流程
    :param file_path: 输入CSV文件路径
    :param transaction_cost: 交易成本（默认0.1%）
    :param risk_free_rate: 无风险利率（年化，默认2%）
    :return: 结果数据框 + 核心指标字典
    """
    # 步骤1：数据预处理
    print("Step 1: Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)
    
    # 步骤2：计算策略收益
    print("\nStep 2: Calculating strategy returns...")
    result_df = calculate_strategy_returns(df, transaction_cost, risk_free_rate)
    
    # 步骤3：计算核心指标
    print("\nStep 3: Calculating core metrics...")
    all_metrics = calculate_core_metrics(result_df=result_df, risk_free_rate=risk_free_rate,stock_code=stock_code)
    print("\nBacktest analysis completed successfully!")
    return result_df, all_metrics

if __name__ == "__main__":
    # 实际文件路径（根据你的环境修改）
    # INPUT_FILE_PATH = 'huiceresultdata/000001.SZ.csv'
    # result_df, backtest_metrics = main(INPUT_FILE_PATH)
    # print(backtest_metrics)
    folder = 'huiceresultdata/'
    result_list=[]
    _ ,metrics=main(folder+"000001.SZ.csv",stock_code="000001.SZ")
    print(metrics)
    for csv_file in os.listdir(folder): 
        INPUT_FILE_PATH = 'huiceresultdata/'+csv_file
    # 执行回测分析
        _, backtest_metrics = main(INPUT_FILE_PATH,stock_code=csv_file[:-4])
        result_list.append(backtest_metrics)
    df=pd.DataFrame(result_list)
    df.to_csv('策略回测结果_多组数据_100天.csv', index=False)
    # # 打印核心指标（重点关注交易相关指标）
    # print("\n=== 核心回测指标 ===")
    # for key, value in backtest_metrics.items():
    #     if 'trade' in key or 'win' in key or 'profit' in key:
    #         print(f"{key}: {value:.4f}")
    # print("\n=== 收益与风险指标 ===")
    # print(f"基准总收益: {backtest_metrics['benchmark_total_return']:.4f}")
    # print(f"策略总收益（扣成本）: {backtest_metrics['strategy_total_return']:.4f}")
    # print(f"策略夏普比率: {backtest_metrics['strategy_sharpe']:.4f}")
    # print(f"策略最大回撤: {backtest_metrics['strategy_max_drawdown']:.4f}")



'''
    指标名称	您的结果数值	通俗含义
benchmark_sharpe	-0.020173117372035105	基准策略（持有不动）的夏普比率，风险性价比较差
strategy_sharpe_raw	-0.010404503532609	VG-GCN 策略（未扣成本）的夏普比率，风险性价比优于基准
strategy_sharpe	-0.010404503532609	VG-GCN 策略（已扣成本）的夏普比率，扣成本后风险性价比未变
二、累计回报率（共 6 个，名称带 “return”）
累计回报率反映 “整个回测期的总收益” 和 “换算成 1 年的年化收益”，负数代表亏损，数值越接近 0（或正数）越好。
指标名称	您的结果数值	通俗含义
benchmark_total_return	-0.08944793850454213	基准策略总收益率：194 天亏损 8.94%
strategy_total_return_raw	-0.04835622276358009	VG-GCN 未扣成本总收益率：194 天亏损 4.84%
strategy_total_return	-0.05681054647025918	VG-GCN 已扣成本总收益率：194 天亏损 5.68%
benchmark_annual_return	-0.11460274668932413	基准策略年化收益率：换算成 1 年亏损 11.46%
strategy_annual_return_raw	-0.062353952908901555	VG-GCN 未扣成本年化收益率：换算成 1 年亏损 6.24%
strategy_annual_return	-0.07315992558197248	VG-GCN 已扣成本年化收益率：换算成 1 年亏损 7.32%
三、损益分析（共 5 个，名称带 “trades/win/profit”）
损益分析聚焦 “交易行为的盈亏细节”，您的结果中这类指标为 0，是因为策略未产生任何买卖操作（total_trades=0）。
指标名称	您的结果数值	通俗含义
total_trades	0	总交易次数：194 天内无任何买卖操作
winning_trades	0	盈利交易次数：无交易→无盈利交易
losing_trades	0	亏损交易次数：无交易→无亏损交易
win_rate	0	胜率（盈利交易占比）：无交易→胜率为 0
profit_factor	0	盈亏比（总盈利 ÷ 总亏损）：无交易→盈亏比为 0

'''

'''
回测结果指标含义全解析（结合您的 194 个交易日数据）
您提供的 194 个交易日回测结果，可分为5 大类核心指标，下面逐个用通俗语言解释含义，并结合您的具体数值说明实际意义：
1. 收益类指标：衡量 “赚 / 亏了多少”
这类指标反映策略和基准（持有不动）的收益能力，负数代表亏损，数值越接近 0（或正数）越好。
指标名称	您的数值	通俗含义	实际解读（194 个交易日）
benchmark_total_return	-0.0894	基准策略（买入后持有不动）的总收益率（小数形式）	持有股票 194 天，最终总亏损 8.94%（-0.0894×100%）
strategy_total_return_raw	-0.0484	VG-GCN 策略未扣除交易成本的总收益率	不考虑手续费 / 印花税时，策略总亏损 4.84%，比基准少亏 4.1%（8.94%-4.84%）
strategy_total_return	-0.0568	VG-GCN 策略扣除交易成本后的总收益率	算上交易手续费后，策略总亏损 5.68%，仍比基准少亏 3.26%（8.94%-5.68%）
benchmark_annual_return	-0.1146	基准策略的年化收益率（换算成 1 年 252 个交易日的收益）	若持有 1 年，基准策略预计亏损 11.46%（反映长期持有风险）
strategy_annual_return_raw	-0.0624	VG-GCN 未扣成本的年化收益率	未扣成本时，策略年化亏损 6.24%，比基准年化少亏 5.22%
strategy_annual_return	-0.0732	VG-GCN 扣成本后的年化收益率	扣成本后，策略年化亏损 7.32%，仍比基准年化少亏 4.14%
2. 风险调整收益指标：衡量 “风险性价比”（夏普比率）
夏普比率是核心的 “风险调整收益指标”，本质是 “每承担 1 单位风险，能获得多少超额收益”，数值越高越好（正＞负，负的越小越好）。
指标名称	您的数值	通俗含义	实际解读
benchmark_sharpe	-0.0202	基准策略的夏普比率	持有不动的 “风险性价比差”：承担 1 单位风险，获得的收益还跑不赢无风险利率（如国债），且亏损效率比策略高
strategy_sharpe_raw	-0.0104	VG-GCN 未扣成本的夏普比率	未扣成本时，策略的 “风险性价比优于基准”：同样承担 1 单位风险，亏损幅度比基准低约 50%（-0.0104 比 - 0.0202 更接近 0）
strategy_sharpe	-0.0104	VG-GCN 扣成本后的夏普比率	扣成本后夏普比率未变，说明交易成本对 “风险性价比” 影响极小（可能因交易次数少）
3. 交易行为与损益分析：衡量 “交易是否有效”
这类指标反映策略的实际交易操作和盈亏结构，您的结果中多数为 0，需重点关注。
指标名称	您的数值	通俗含义	实际解读（关键！）
total_trades	0	策略总交易次数（1 次买入 + 1 次卖出算 1 笔完整交易）	您的策略在 194 天内没有产生任何买卖操作（可能是策略信号signal始终为 0，未触发买入 / 卖出）
winning_trades	0	盈利的交易次数	无交易→无盈利交易
losing_trades	0	亏损的交易次数	无交易→无亏损交易
win_rate	0	盈利交易占总交易的比例（胜率）	无交易→胜率为 0（正常交易下，胜率＞50% 更优）
profit_factor	0	总盈利金额 ÷ 总亏损金额（盈亏比）	无交易→盈亏比为 0（正常交易下，盈亏比＞1 更优，说明赚的钱能覆盖亏的钱）
4. 风险控制指标：衡量 “最大下跌风险”（最大回撤）
最大回撤是 “策略从历史最高点到最低点的最大跌幅”，负数绝对值越小，风险控制越好（代表下跌时亏得更少）。
指标名称	您的数值	通俗含义	实际解读
benchmark_max_drawdown	-0.3703	基准策略的最大回撤	持有不动期间，股票从最高点跌到最低点，最大亏了 37.03%（风险较高）
strategy_max_drawdown	-0.3463	VG-GCN 扣成本后的最大回撤	策略期间，最大跌幅仅 34.63%，比基准少跌 2.4%，风险控制更优
strategy_max_drawdown_raw	-0.3411	VG-GCN 未扣成本的最大回撤	不扣成本时，最大跌幅 34.11%，风险控制效果更好
5. 基础数据维度：回测的 “时间范围”
指标名称	您的数值	通俗含义	实际解读
trading_days	194	回测覆盖的实际交易日数量	约等于 10 个月交易时间（按每月 20 个交易日估算），回测周期有一定参考价值
整体结论（结合所有指标）
收益优势：VG-GCN 策略无论是否扣成本，都比 “持有不动” 亏得少（总亏损少 3.26%-4.1%，年化亏损少 4.14%-5.22%），在熊市环境下表现更抗跌。
风险优势：策略最大回撤（34.11%-34.63%）低于基准（37.03%），风险控制能力更强。
关键问题：策略未产生任何交易（total_trades=0），可能是signal列未输出有效的 “买入（1）/ 卖出（-1）” 信号，建议检查原始数据中signal的取值（是否全为 0）。

'''
