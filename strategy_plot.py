import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def plot_strategy_comparison(file_path, save_path='/mnt/strategy_comparison_chart.png'):
    """
    绘制股票不同投资策略收益对比折线图
    
    参数:
    file_path: CSV文件路径
    save_path: 图片保存路径
    """
    
    # 设置中文字体和图表样式
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # 读取并预处理数据
    df = pd.read_csv(file_path)
    
    # 数据有效性检查
    required_columns = ['日期（YYYY-MM-DD）', 'Buy&Hold_Strategy（元）', 
                       'Strategy_no_cost（元）', 'Strategy_10bps（元）']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据文件缺少必要列: {col}")
    
    # 将日期列转换为日期格式
    df['日期（YYYY-MM-DD）'] = pd.to_datetime(df['日期（YYYY-MM-DD）'])
    
    # 创建专业的折线图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 定义颜色、线型和标签
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 红色、青色、蓝色
    linestyles = ['-', '--', '-.']  # 实线、虚线、点划线
    labels = ['买入持有策略', '无成本策略', '10bps成本策略']
    columns = ['Buy&Hold_Strategy（元）', 'Strategy_no_cost（元）', 'Strategy_10bps（元）']
    
    # 绘制三条折线
    for i, (col, color, linestyle, label) in enumerate(zip(columns, colors, linestyles, labels)):
        ax.plot(df['日期（YYYY-MM-DD）'], df[col], 
                color=color, 
                linestyle=linestyle, 
                label=label, 
                linewidth=2.5,
                alpha=0.8,
                marker='o',  # 标记点
                markersize=4,
                markevery=5)  # 每5个点显示一个标记
    
    # 设置图表标题和标签
    ax.set_title('000301.SZ 股票不同投资策略收益对比', 
                 fontsize=18, 
                 fontweight='bold', 
                 pad=20,
                 color='#2C3E50')
    
    ax.set_xlabel('日期', fontsize=14, fontweight='bold', labelpad=10, color='#34495E')
    ax.set_ylabel('策略收益（元）', fontsize=14, fontweight='bold', labelpad=10, color='#34495E')
    
    # 设置x轴日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 每2个月显示一个刻度
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    
    # 设置y轴范围，留出适当边距
    y_min = df[columns].min().min() * 0.95
    y_max = df[columns].max().max() * 1.05
    ax.set_ylim(y_min, y_max)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#BDC3C7')
    
    # 添加图例
    legend = ax.legend(loc='upper right', 
                      fontsize=12, 
                      frameon=True, 
                      fancybox=True, 
                      shadow=True, 
                      framealpha=0.9,
                      facecolor='white',
                      edgecolor='#EAECEE')
    legend.get_frame().set_linewidth(0.5)
    
    # 添加数值标注（起点和终点）
    for i, (col, color, label) in enumerate(zip(columns, colors, labels)):
        # 起点标注
        start_x = df['日期（YYYY-MM-DD）'].iloc[0]
        start_val = df[col].iloc[0]
        ax.annotate(f'{start_val:.2f}', 
                    xy=(start_x, start_val),
                    xytext=(10, 5),
                    textcoords='offset points',
                    fontsize=10,
                    color=color,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color, linewidth=0.5))
        
        # 终点标注
        end_x = df['日期（YYYY-MM-DD）'].iloc[-1]
        end_val = df[col].iloc[-1]
        ax.annotate(f'{end_val:.2f}', 
                    xy=(end_x, end_val),
                    xytext=(10, 5),
                    textcoords='offset points',
                    fontsize=10,
                    color=color,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color, linewidth=0.5))
    
    # 添加水平参考线（初始值1.00）
    ax.axhline(y=1.00, color='#95A5A6', linestyle=':', linewidth=1.5, alpha=0.7, label='初始值')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（高分辨率）
    # plt.savefig(save_path, 
    #             dpi=300, 
    #             bbox_inches='tight', 
    #             facecolor='white', 
    #             edgecolor='none')
    plt.show()
    plt.close()
    
    # 输出数据统计信息
    print("="*80)
    print("000301.SZ 股票不同投资策略收益统计")
    print("="*80)
    print(f"数据时间范围: {df['日期（YYYY-MM-DD）'].min().strftime('%Y-%m-%d')} 至 {df['日期（YYYY-MM-DD）'].max().strftime('%Y-%m-%d')}")
    print(f"数据点数: {len(df)} 个")
    print()
    
    for col, label in zip(columns, labels):
        start = df[col].iloc[0]
        end = df[col].iloc[-1]
        change = ((end - start) / start) * 100
        max_val = df[col].max()
        min_val = df[col].min()
        vol = ((max_val - min_val) / start) * 100  # 波动率
        
        print(f"{label}:")
        print(f"  起始值: {start:.2f} 元")
        print(f"  结束值: {end:.2f} 元")
        print(f"  总收益率: {change:+.2f}%")
        print(f"  最高值: {max_val:.2f} 元")
        print(f"  最低值: {min_val:.2f} 元")
        print(f"  区间波动率: {vol:.2f}%")
        print()
    
    print(f"图表已保存至: {save_path}")
    return df

# 主程序
if __name__ == "__main__":
    STOCK_CODE="000065.SZ"
    # 示例用法
    file_path = '权益曲线/'+STOCK_CODE+'.csv'  # 输入文件路径
    save_path = '/mnt/strategy_comparison_chart.png'  # 输出图片路径
    
    try:
        # 执行绘图函数
        df_result = plot_strategy_comparison(file_path, save_path)
        print("\n绘图完成！")
    except Exception as e:
        print(f"绘图过程中出现错误: {str(e)}")
