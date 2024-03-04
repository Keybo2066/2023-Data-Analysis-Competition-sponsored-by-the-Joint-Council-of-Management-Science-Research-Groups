import lttb
import pickle
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import mlab
from statsmodels.tsa.seasonal import STL, MSTL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels import api as sm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import operator
import math
from scipy import signal
from tqdm import tqdm
import copy
from scipy.signal import find_peaks
import matplotlib.animation as animation

base = Path(__file__).resolve().parent
with open(os.path.join(base, 'time_series_by_genres.pkl'), 'rb') as f:
    time_series = pickle.load(f)

plt.rcParams['font.family'] = "MS Gothic"

ATRS = ['observed', 'trend', 'seasonal', 'resid']
DFT_CYCLE = 13
COST =20000000
BUDGET = 1e8
# BUDGET = 1e100
# (x,y)データを作成
x = np.arange(50, 300)
y = np.arange(0.5,2.5, 0.01)

# 格子点の作成
X, Y = np.meshgrid(x, y)


def plot(df):
    df.plot()
    plt.xticks(np.arange(9, 105, step=DFT_CYCLE))
    plt.grid(axis='x')
    plt.show()


def stl(df):
    stl = STL(df, period=DFT_CYCLE, robust=True)
    stl_series = stl.fit()
    stl_series.plot()
    plt.show()


def seasonal_decomp(df, model, show):
    # STL実行とSTLの結果をライブラリのメソッドそのまま利用で表示

    model = 'additive' if isAdd else 'multiplicative'
    decomposed_data = sm.tsa.seasonal_decompose(
        df, model=model, extrapolate_trend='freq', period=DFT_CYCLE)
    if not show:
        return decomposed_data
    decomposed_data.plot()
    plt.xticks(np.arange(9, 105, step=DFT_CYCLE))
    plt.grid(axis='x')
    plt.show()


def addormult(changed_data, model):
    # 加算/乗算モデルに基づくchangede_dataの復元

    # STL分解データが渡されたらアトリビューションを走査してlistに変換
    if not isinstance(changed_data, list):
        atrs = ['trend', 'seasonal', 'resid'] if len(changed_data) == 4 else [
            'trend', 'seasonal_literal', 'seasonal_seal', 'resid']
        changed_data = np.array([getattr(changed_data, atr) for atr in atrs])
    result = np.sum(changed_data, axis=0) if model == 'additive' else np.prod(
        changed_data, axis=0)
    return result



def stl_decomp_change_seasonal(df, model, magnification, strength):
    # STLを実行しseasonalを変化させる
    # 古典的分解の手法になっている

    class Dict_changed_data(dict):  # dotアクセスするためにdict継承でクラス作成
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    # model='additive' or 'multiplicative'
    decomposed_data = sm.tsa.seasonal_decompose(
        df, model=model, extrapolate_trend='freq', period=DFT_CYCLE)

    # 分解されたデータはpdなのでnpに変換
    dict_decomposed_data = dict(
        zip(ATRS, map(lambda x: getattr(decomposed_data, x).values, ATRS)))
    decomposed_data = Dict_changed_data(dict_decomposed_data)

    peak = decomposed_data.seasonal[:DFT_CYCLE]  # ピークの一山だけ抽出
    # peak = decomposed_data.seasonal
    resampled_data = signal.resample(peak, magnification)  # resample

    if strength:
        # 非線形に変形
        map_func = np.frompyfunc(
            lambda x, a: x-1/a*((a*x+1)+1/(a*x+1)-2), 2, 1)
        min_sale = np.min(resampled_data)
        resampled_data -= min_sale  # 最小を0に平行移動
        resampled_data = map_func(resampled_data, strength)
        resampled_data += min_sale  # 元に戻す

    # 繰り返して105で切る
    resampled_data = np.tile(
        resampled_data, math.ceil(105/len(resampled_data)))[:105]

    changed_data = Dict_changed_data(observed=decomposed_data.observed,
                                     trend=decomposed_data.trend,
                                     seasonal=resampled_data,
                                     resid=decomposed_data.resid)

    return decomposed_data, changed_data


def mstl_decomp_change_seasonal(df, model, magnification, strength):
    # MSTLを実行しseasonalを変化させる

    class Dict_changed_data(dict):  # dotアクセスするためにdict継承でクラス作成
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    # model='additive' or 'multiplicative'
    decomposed_data = MSTL(df, periods=[4, 13], stl_kwargs={
                           "seasonal_deg": 0}).fit()

    # 分解されたデータはpdなのでnpに変換
    dict_decomposed_data = dict(
        zip(ATRS, map(lambda x: getattr(decomposed_data, x).values, ATRS)))
    decomposed_data = Dict_changed_data(dict_decomposed_data)

    # seasonal = copy.copy(
    #     decomposed_data.seasonal[:, 1][:DFT_CYCLE])  # ピークの一山だけ抽出

    seasonal=copy.copy(decomposed_data.seasonal[:, 1])


    resampled_seasonal_data = signal.resample(
        seasonal, magnification)  # resample
    # 繰り返して105で切る
    resampled_data = np.tile(
        resampled_seasonal_data, math.ceil(105/len(resampled_seasonal_data)))[:105]

    #maxを元データと合わせる
    map_func = np.frompyfunc(lambda x, a: x-1/a*((a*x+1)+1/(a*x+1)-2), 2, 1)
    alpha = 1/max(decomposed_data.seasonal[:, 1]) - 1/max(resampled_data)
    if alpha:
        resampled_data = map_func(resampled_data, alpha)

    # 非線形に変形
    min_sale = np.min(resampled_data)
    resampled_data -= min_sale  # 最小を0に平行移動
    # map_func = np.frompyfunc(
    #     lambda x, a: x-1/a*((a*x+1)+1/(a*x+1)-2), 2, 1)
    alpha = 1/max(resampled_data)/strength - 1/max(resampled_data)
    if alpha:
        resampled_data = map_func(resampled_data, alpha)
    # resampled_data *= strength
    resampled_data += min_sale  # 元に戻す

    changed_data = Dict_changed_data(observed=decomposed_data.observed,
                                     trend=decomposed_data.trend,
                                     seasonal_literal=decomposed_data.seasonal[:, 0],
                                     seasonal_seal=resampled_data,
                                     resid=decomposed_data.resid)

    decomposed_data_flat = Dict_changed_data(observed=decomposed_data.observed,
                                             trend=decomposed_data.trend,
                                             seasonal_literal=decomposed_data.seasonal[:, 0],
                                             seasonal_seal=decomposed_data.seasonal[:, 1],
                                             resid=decomposed_data.resid)

    return decomposed_data_flat, changed_data


def sub_cost_calc_ratio(decomposed_data, changed_data, model, magnification, strength, return_cost=False):
    # コストを考慮した純売上を計算する
    seasonal_atr = 'seasonal'if len(changed_data) == 4 else 'seasonal_seal'
    decomposed_result = addormult(decomposed_data, model)
    changed_result = addormult(changed_data, model)
    peaks, _ = find_peaks(getattr(changed_data, seasonal_atr),height=5e6,distance=6)
    net_decomposed_sales = sum(decomposed_result) - COST*8

    # after_cost = COST*len(peaks)*pow(strength,6)
    after_cost = COST*len(peaks)*strength
    net_changed_sales = sum(changed_result) - after_cost
    if return_cost:
        return net_changed_sales/net_decomposed_sales, after_cost

    else:
        if after_cost <= BUDGET:
            return net_changed_sales/net_decomposed_sales
        else:
            return np.nan
    # return net_changed_sales/net_decomposed_sales
    # return COST*len(peaks)*pow(strength,6)


def calc_sales(genres, model, magnification, strength, pbar=None):
    # データから比を出すまでの過程をまとめる
    if genre is str:
        data = time_series[genres]
    else:
        data = genres

    decomposed_data, changed_data = mstl_decomp_change_seasonal(
        data, model, magnification, strength)
    if pbar:
        pbar.update(1)
    return sub_cost_calc_ratio(decomposed_data, changed_data, model, magnification, strength, return_cost=True)


def seasonal_show(decomposed_data, changed_data, model, magnification):
    # (M)STL分解とchanged_dataを並列して描画

    seasonal_atr = 'seasonal'if len(changed_data) == 4 else 'seasonal_seal'
    peaks, _ = find_peaks(getattr(changed_data, seasonal_atr),height=5e6,distance=6)
    decomposed_result = addormult(decomposed_data, model)
    changed_result = addormult(changed_data, model)

    row, col = len(changed_data)+1, 2
    fig, axes = plt.subplots(row, col, squeeze=False, tight_layout=True)
    for i in range(row):  # セールタイミングを線で表示
        for j in range(col):
            if j==0:
                axes[i, j].set_xticks(np.arange(9, 105, step=DFT_CYCLE))
            else:
                axes[i, j].set_xticks(peaks)
            axes[i, j].grid(axis='x')

    atrs = ['observed', 'trend', 'seasonal_literal',
            'seasonal_seal', 'resid'] if row == 6 else ATRS

    for i, atr in enumerate(atrs):
        axes[i, 0].plot(getattr(decomposed_data, atr))
        axes[i, 1].plot(getattr(changed_data, atr))
    axes[row-1, 0].plot(decomposed_result)
    axes[row-1, 1].plot(changed_result)

    # y軸レンジを合わせる
    isBigDecomp = np.max(getattr(decomposed_data, seasonal_atr)) > np.max(
        getattr(changed_data, seasonal_atr))
    plus_row = seasonal_atr == 'seasonal_seal'
    for i in [2+plus_row, 4+plus_row]:
        axes[i, int(isBigDecomp)].set_ylim(
            *axes[i, int(not isBigDecomp)].get_ylim())
    plt.show()


def double(df):
    fig, axes = plt.subplots(4, 2, tight_layout=True)
    for i in range(2):
        for j, atr in enumerate(ATRS):
            axes[j, i].plot(
                getattr(seasonal_decomp(df, not bool(i), False), atr))
            axes[j, i].set_title(f"{not bool(i)}")
    plt.show()


def acor(df):
    plot_acf(df, lags=29)
    plt.axvline(x=DFT_CYCLE)
    plt.show()





def loop_strength(max_range):
    results = []
    range_st = [num * 0.01 for num in range(1, max_range)]
    for strength in range_st:
        decomposed_data, changed_data = mstl_decomp_change_seasonal(
            time_series['ファッション'], model, magnification, strength)
        # seasonal_show(decomposed_data, changed_data, model, magnification)
        c = sub_cost_calc_ratio(decomposed_data,
                                changed_data, model, magnification, strength)
        results.append(c)
    print((results.index(max(results))-max_range)*0.01)
    plt.plot(pd.Series(results, index=range_st))
    plt.show()


def loop_magnification(max_range):
    results = []
    range_st = list(range(40, max_range))
    for magnification in range_st:
        decomposed_data, changed_data = mstl_decomp_change_seasonal(
            time_series['ファッション'], model, magnification, strength)
        # seasonal_show(decomposed_data, changed_data, model, magnification)
        results.append(sub_cost_calc_ratio(decomposed_data,
                                           changed_data, model, magnification, strength))
    print(results.index(max(results)))
    plt.plot(pd.Series(results, index=range_st))
    plt.show()


def calc_Z(genre):

    np_calc_sales = np.frompyfunc(calc_sales, 5, 2)

    # 売上の計算式
    with tqdm(total=len(X)*len(X[0]),leave=False) as pbar:
        # my_output = np.vectorize(my_function)(my_inputs, pbar)
        Z,after_cost = np_calc_sales(genre, model, X, Y, pbar)
    # max_index = np.unravel_index(np.argmax(Z), Z.shape)
    return X,Y,Z,after_cost


def tri_plot(X,Y,Z):

    # Figureを追加
    fig = plt.figure(figsize = (10, 6))

    # 3DAxesを追加
    ax = fig.add_subplot(111, projection="3d")

    # ワイヤーフレームを描く
    ax.plot_surface(X, Y, Z, cmap = "plasma_r")
    max_x_index,max_y_index = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax.scatter(x[max_y_index],y[max_x_index],np.nanmax(Z),c='red')
    print(x[max_y_index],y[max_x_index],np.nanmax(Z))


    plt.show()


def rotate3d(X,Y,Z,genre):
    from IPython.display import HTML


    # Figureを追加
    fig = plt.figure(figsize = (10, 6))

    # 3DAxesを追加
    ax = fig.add_subplot(111, projection="3d")

    def init():
        ax.plot_surface(X, Y, Z, cmap = "plasma_r")
        max_x_index,max_y_index = np.unravel_index(np.nanargmax(Z), Z.shape)
        ax.scatter(x[max_y_index],y[max_x_index],np.nanmax(Z),c='red')
        ax.set_title(genre)
        ax.set_xlabel('セール頻度')
        ax.set_ylabel('セール強度')
        ax.set_zlabel('通常売上に対する売上比')
        return fig,

    def animate(i):
        ax.view_init(elev=30., azim=3.6*i)
        return fig,

    # Animate
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=100, interval=100, blit=True)
    return ani

    # HTML(ani.to_html5_video())

model = 'additive'
# model = 'multiplicative'

magnification = 105  # 105基準
strength = 1  # 1基準

def single(magnification,strength,genre):
    decomposed_data, changed_data = mstl_decomp_change_seasonal(
        time_series[genre], model, magnification, strength)
    seasonal_show(decomposed_data, changed_data, model, magnification)
    print(sub_cost_calc_ratio(decomposed_data, changed_data, model, magnification,strength))

def pickle_Z():
    genres = time_series.keys()
    Zs = {}
    costs = {}
    for genre in tqdm(genres):
        X,Y,Z,after_cost = calc_Z(genre)
        Zs[genre] = Z
        costs[genre] = after_cost
        # tri_plot(X, Y, Z)
        # with open(f'pickle/{genre}.pkl', 'wb') as f:
        #     pickle.dump(Z, f)
        # with open(f'pickle/{genre}_cost.pkl', 'wb') as f:
        #     pickle.dump(after_cost, f)
        Z = [[z if c <= BUDGET else np.nan for z,c in zip(z_row,c_row)] for z_row,c_row in zip(Z,after_cost)]
        Z = np.array(Z)
        ani = rotate3d(X,Y,Z,genre)
        ani.save(f'3dwf/{genre}_3dwf.mp4', writer="ffmpeg",dpi=100)

    with open(f'Zs.pkl', 'wb') as f:
        pickle.dump(Zs, f)
    with open(f'costs.pkl', 'wb') as f:
        pickle.dump(costs, f)

def all_genre_sales_3dplot(data):
    X,Y,Z,after_cost = calc_Z(data)
    Z = [[z if c <= BUDGET else np.nan for z,c in zip(z_row,c_row)] for z_row,c_row in zip(Z,after_cost)]
    Z = np.array(Z)
    ani = rotate3d(X,Y,Z,genre)
    ani.save(f'3dwf/{genre}_3dwf.mp4', writer="ffmpeg",dpi=100)


# 関数all_genre_sales_3dplotに総売上のデータを渡せば3dplotの動画が作成される


# with open(f'pickle/ファッション.pkl', 'rb') as f:
#     Z = pickle.load(f)
# with open(f'pickle/ファッション_cost.pkl', 'rb') as f:
#     cost = pickle.load(f)


# Z = [[z if c <= BUDGET else np.nan for z,c in zip(z_row,c_row)] for z_row,c_row in zip(Z,cost)]
# Z = np.array(Z)
# tri_plot(X,Y,Z)
# tri_plot(X, Y, cost)
# pickle_Z()
# single(89,1.2,'ファッション')
# loop_magnification(700)
# loop_strength(300)

# with open(f'Zs.pkl', 'rb') as f:
#     Zs = pickle.load(f)
# with open(f'costs.pkl', 'rb') as f:
#     costs = pickle.load(f)


# # print(Zs)
# for Z,cost in zip(Zs.values(),costs.values()):
#     # Z = [[z if c <= BUDGET else np.nan for z,c in zip(z_row,c_row)] for z_row,c_row in zip(Z,cost)]
#     # Z = np.array(Z)
#     # tri_plot(X,Y,Z)
#     # print(Z)
#     ani = rotate3d(X,Y,Z)
#     ani.save(f'test.mp4', writer="ffmpeg",dpi=100)
# for g in time_series.values():
#     print(g)

# Z = Zs['グルメ・飲料']
# ani = rotate3d(X,Y,Z)
# ani.save(f'test.mp4', writer="ffmpeg",dpi=100)

# decomposed_data, changed_data = mstl_decomp_change_seasonal(
#     time_series['ファッション'], model, magnification, strength)
# resampled_data = copy.copy(changed_data['seasonal_seal'])
# plt.plot(resampled_data,label=f'strength=1.0')

# y_down = signal.resample(y, 89)
# x_down = np.linspace(0, 104, len(y_down))
# # y_down = np.tile(
# #         y_down, math.ceil(105/len(y_down)))[:105]
# plt.plot(x_down, y_down, 's-',label='down-sampled', )
# plt.plot(y,'o-',alpha=0.7,label='data')
# plt.legend()
# # plt.plot(y_down,'s-')
# plt.xticks([0,104])
# plt.grid(axis='x')
# plt.show()

# def temp(strength,resampled_data2):
#     map_func = np.frompyfunc(lambda x, a: x-1/a*((a*x+1)+1/(a*x+1)-2), 2, 1)

#     # 非線形に変形
#     min_sale = np.min(resampled_data2)
#     print(min_sale)
#     resampled_data2 -= min_sale  # 最小を0に平行移動
#     # map_func = np.frompyfunc(
#     #     lambda x, a: x-1/a*((a*x+1)+1/(a*x+1)-2), 2, 1)
#     alpha = 1/max(resampled_data2)/strength - 1/max(resampled_data2)
#     if alpha:
#         resampled_data2 = map_func(resampled_data2, alpha)
#     # resampled_data2 *= strength
#     resampled_data2 += min_sale  # 元に戻す
#     plt.plot(resampled_data2,'o-',label=f'{strength=}',alpha=0.7)
# temp(1.0,resampled_data)
# temp(1.2,resampled_data)
# resampled_data3 = copy.copy(changed_data['seasonal_seal'])
# temp(0.8,resampled_data3)
# plt.legend()
# # plt.ylim(top = max(changed_data['seasonal_seal']),bottom=min(changed_data['seasonal_seal']))
# plt.show()

