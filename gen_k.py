import pandas as pd
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import mplfinance as mpf

# 读取 CSV 数据
EUA = pd.read_csv(r"F:\Futures_Quant\全-数值型-大豆、豆粕、豆油\全-数值型-大豆、豆粕、豆油\豆油.csv", encoding='gbk')

# 确保日期列是 datetime 格式
EUA['date'] = pd.to_datetime(EUA['date'])

# 创建输出文件夹（如果不存在的话）
output_dir = r"F:\Futures_Quant\gen_K_photo\soy_oil_pic"
os.makedirs(output_dir, exist_ok=True)

# 定义绘制 K 线图的函数
def generate_k_line_chart(i):
    if i + 4 < len(EUA):  # 确保我们不会超出数据的索引范围
        seg_EUA = EUA.iloc[i:i+5]  # 取出连续5天的数据
        seg_EUA.set_index('date', inplace=True)  # 将日期列设置为索引

        # 设置自定义市场颜色
        market_colors = mpf.make_marketcolors(up='r', down='g', edge='inherit', wick='inherit')

        # 定义自定义的绘图风格，使用 `yahoo` 基础样式并禁用所有网格线
        my_style = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=market_colors)

        # 使用 `addplot` 来绘制K线图，并且禁用所有网格线和坐标轴
        fig, ax = mpf.plot(
            seg_EUA, 
            type='candle', 
            style=my_style, 
            ylabel='',  # 不显示y轴标签
            returnfig=True  # 返回figure对象
        )

        # 调整边距，使K线图占据整个图片
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # 禁用所有网格线和坐标轴
        ax[0].grid(False)                  # 禁用网格线
        ax[0].spines['left'].set_visible(False)  # 移除左边框
        ax[0].spines['right'].set_visible(False) # 移除右边框
        ax[0].spines['top'].set_visible(False)   # 移除顶部边框
        ax[0].spines['bottom'].set_visible(False) # 移除底部边框
        ax[0].xaxis.set_visible(False)  # 隐藏x轴
        ax[0].yaxis.set_visible(False)  # 隐藏y轴

        # 保存图像，使用bbox_inches='tight'和pad_inches=0来移除多余的空白区域
        fig.savefig(
            os.path.join(output_dir, f"{seg_EUA.index[0].strftime('%Y-%m-%d')}_to_{seg_EUA.index[4].strftime('%Y-%m-%d')}_no_grid.jpg"),
            bbox_inches='tight', pad_inches=0
        )
        plt.close(fig)


# 使用多进程来加速图表生成
if __name__ == '__main__':
    with Pool() as pool:
        pool.map(generate_k_line_chart, range(len(EUA) - 2))

    print("Done!")
