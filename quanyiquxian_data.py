import pandas as pd
import numpy as np

# -------------------------- 1. 基础参数配置（可根据需求调整） --------------------------
INITIAL_CAPITAL = 1 # 初始资金：100万元（金融回测常规值）
TRANSACTION_COST = 0.001   # 10bps策略成本：0.1%（10bps=0.1%），无成本策略此值设为0
STOCK_CODE = "000065.SZ"   # 股票代码
TARGET_DAYS = 100          # 目标生成“后100天”权益数据
# 请根据你的CSV文件，修改以下列名（必须匹配！）
DATE_COL = "日期"           # 日期列名（如你的CSV列名是“trade_date”，需修改此处）
CLOSE_COL = "收盘价"        # 实际每日收盘价列名（用于计算真实收益）
PREDICT_COL = "预测值"      # 预测值列名（1=涨/持平，0=跌，预测次日涨跌）
OUT_PATH="权益曲线/"+STOCK_CODE+".csv"
# -------------------------- 2. 数据读取与预处理 --------------------------
# 读取预测结果CSV
df = pd.read_csv("huiceresultdata/"+STOCK_CODE+".csv")

# 1. 检查关键列是否存在（避免列名不匹配报错）
required_cols = [DATE_COL, CLOSE_COL, PREDICT_COL]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"你的CSV文件缺少关键列：{missing_cols}\n请修改代码中【基础参数配置】的列名，确保与CSV一致！")

# 2. 数据清洗：删除预测值/收盘价/日期为空的行
df = df.dropna(subset=[DATE_COL, CLOSE_COL, PREDICT_COL])
# 确保预测值仅为1或0（过滤异常值）
df = df[df[PREDICT_COL].isin([0, 1])].reset_index(drop=True)

# 3. 日期标准化与排序（核心：确保时间顺序正确）
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")  # 转换为datetime格式
df = df.dropna(subset=[DATE_COL])  # 删除日期转换失败的行
df_sorted = df.sort_values(by=DATE_COL).reset_index(drop=True)  # 按日期升序排序

# 4. 筛选“后100天”的有效数据（需包含预测值对应的次日收盘价）
# 逻辑：预测t日的“预测值”对应t+1日的涨跌，因此需取最后101行（t0-t100），计算t1-t100（100天）的权益
if len(df_sorted) < TARGET_DAYS + 1:
    raise ValueError(f"有效数据仅{len(df_sorted)}行，不足{TARGET_DAYS+1}行（需包含预测值对应的次日收盘价）\n请补充更多预测结果数据！")
# 取最后101行数据（t0-t100），用于计算t1-t100（100天）的权益
df_strategy = df_sorted.tail(TARGET_DAYS + 1).reset_index(drop=True)
# 提取后100天的日期（t1-t100，作为最终权益曲线的时间轴）
strategy_dates = df_strategy[DATE_COL].iloc[1:].dt.strftime("%Y-%m-%d").tolist()

print(f"数据预处理完成！")
print(f"策略回测时间范围：{strategy_dates[0]} 至 {strategy_dates[-1]}（共{TARGET_DAYS}天）")
print(f"初始资金：{INITIAL_CAPITAL:.2f}元，10bps策略交易成本：{TRANSACTION_COST*100:.2f}%")

# -------------------------- 3. 定义三大策略的权益计算逻辑 --------------------------
def calculate_equity_curves(df, initial_capital, transaction_cost):
    """
    计算三大策略的权益曲线：
    - Buy&Hold：买入后持有，无交易
    - Strategy_no_cost：按预测值交易，无手续费
    - Strategy_10bps：按预测值交易，含10bps手续费
    """
    # 初始化各策略的每日权益列表（索引0对应t0，索引1-100对应t1-t100）
    bh_equity = [initial_capital]    # Buy&Hold权益
    nc_equity = [initial_capital]    # 无成本策略权益
    tb_equity = [initial_capital]    # 10bps策略权益
    
    # 初始化持仓状态（0=现金，1=股票；预测策略初始为现金，Buy&Hold初始买入）
    nc_position = 0  # 无成本策略持仓
    tb_position = 0  # 10bps策略持仓
    # Buy&Hold：t1日全仓买入股票（股数=初始资金/t1日收盘价）
    t1_close = df[CLOSE_COL].iloc[1]  # t1日收盘价
    bh_shares = initial_capital / t1_close  # Buy&Hold持有的股数（允许 fractional shares）
    
    # 循环计算t1-t100（共100天）的权益（i从1到100，对应t1到t100）
    for i in range(1, TARGET_DAYS + 1):
        # 当前日（ti）的关键数据
        current_close = df[CLOSE_COL].iloc[i]    # ti日收盘价
        prev_predict = df[PREDICT_COL].iloc[i-1] # t(i-1)日的预测值（预测ti日涨跌）
        prev_nc_equity = nc_equity[i-1]          # 无成本策略前一日权益
        prev_tb_equity = tb_equity[i-1]          # 10bps策略前一日权益
        
        # -------------------------- 1. Buy&Hold策略权益计算 --------------------------
        # 逻辑：买入后持有，权益=持有的股数×当日收盘价
        current_bh_equity = bh_shares * current_close
        bh_equity.append(current_bh_equity)
        
        # -------------------------- 2. 无成本策略（Strategy_no_cost）权益计算 --------------------------
        # 逻辑：预测1（涨/持平）→ 持股；预测0（跌）→ 空仓（现金，收益0）
        if prev_predict == 1:
            # 预测涨：需持有股票（若前一日是空仓，今日买入）
            if nc_position == 0:
                # 空仓→买入：全仓买入股票（现金→股票，无成本）
                nc_shares = prev_nc_equity / current_close
                nc_position = 1  # 持仓变为股票
                current_nc_equity = nc_shares * current_close  # 权益=股票市值
            else:
                # 已持股：权益随收盘价变化
                nc_shares = prev_nc_equity / df[CLOSE_COL].iloc[i-1]  # 前一日股数
                current_nc_equity = nc_shares * current_close
        else:
            # 预测跌：需空仓（若前一日是持股，今日卖出）
            if nc_position == 1:
                # 持股→卖出：全仓卖出股票（股票→现金，无成本）
                nc_shares = prev_nc_equity / df[CLOSE_COL].iloc[i-1]
                current_nc_equity = nc_shares * current_close  # 权益=现金
                nc_position = 0  # 持仓变为现金
            else:
                # 已空仓：权益不变（现金无收益）
                current_nc_equity = prev_nc_equity
        nc_equity.append(current_nc_equity)
        
        # -------------------------- 3. 10bps成本策略（Strategy_10bps）权益计算 --------------------------
        # 逻辑：与无成本策略一致，但每次交易（买入/卖出）扣除0.1%手续费
        if prev_predict == 1:
            # 预测涨：需持有股票
            if tb_position == 0:
                # 空仓→买入：全仓买入，扣除手续费
                available_capital = prev_tb_equity * (1 - transaction_cost)  # 扣除手续费后的可用资金
                tb_shares = available_capital / current_close  # 实际能买的股数
                tb_position = 1
                current_tb_equity = tb_shares * current_close  # 权益=股票市值
            else:
                # 已持股：权益随收盘价变化（无交易，无成本）
                tb_shares = prev_tb_equity / df[CLOSE_COL].iloc[i-1]
                current_tb_equity = tb_shares * current_close
        else:
            # 预测跌：需空仓
            if tb_position == 1:
                # 持股→卖出：全仓卖出，扣除手续费
                tb_shares = prev_tb_equity / df[CLOSE_COL].iloc[i-1]
                sell_proceeds = tb_shares * current_close  # 卖出总收入
                current_tb_equity = sell_proceeds * (1 - transaction_cost)  # 扣除手续费后的现金
                tb_position = 0
            else:
                # 已空仓：权益不变
                current_tb_equity = prev_tb_equity
        tb_equity.append(current_tb_equity)
    
    # 返回后100天的权益数据（去掉t0的初始值，保留t1-t100）
    return bh_equity[1:], nc_equity[1:], tb_equity[1:]

# -------------------------- 4. 计算三大策略权益曲线 --------------------------
bh_equity_curve, nc_equity_curve, tb_equity_curve = calculate_equity_curves(
    df_strategy, INITIAL_CAPITAL, TRANSACTION_COST
)

# -------------------------- 5. 整理输出数据（适配Origin绘图） --------------------------
# 构建最终输出的DataFrame（时间+三大策略权益，保留2位小数）
output_df = pd.DataFrame({
    "日期（YYYY-MM-DD）": strategy_dates,
    "Buy&Hold_Strategy（元）": [round(eq, 2) for eq in bh_equity_curve],
    "Strategy_no_cost（元）": [round(eq, 2) for eq in nc_equity_curve],
    "Strategy_10bps（元）": [round(eq, 2) for eq in tb_equity_curve]
})

# -------------------------- 6. 保存为CSV文件（输出到/mnt目录） --------------------------
# output_path = f"/mnt/{STOCK_CODE}_后100天三大策略权益曲线.csv"
output_path=OUT_PATH
output_df.to_csv(output_path, index=False, encoding="utf-8-sig")  # utf-8-sig确保中文正常显示

# -------------------------- 7. 输出结果预览 --------------------------
print(f"\n✅ 权益曲线CSV文件已生成！")
print(f"文件路径：{output_path}")
print(f"\n前5天权益数据预览：")
print(output_df.head())
print(f"\n后5天权益数据预览：")
print(output_df.tail())
print(f"\n策略最终权益对比：")
print(f"Buy&Hold策略：{bh_equity_curve[-1]:.2f}元")
print(f"无成本策略：{nc_equity_curve[-1]:.2f}元")
print(f"10bps成本策略：{tb_equity_curve[-1]:.2f}元")