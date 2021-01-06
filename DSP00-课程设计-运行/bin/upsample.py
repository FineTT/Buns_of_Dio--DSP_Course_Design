import numpy as np
import scipy.io.wavfile as wavfile
from sys import argv
from scipy import signal
from scipy import interpolate
from scipy import spatial

def FIR_upsample_filter(fs_x,fs_up,x,x_len_up,up_factor):
    '''
    FIR滤波器增采样处理函数，输入低采样率的信号 x，输出通过滤波方法得到的增采样信号 x_filter。
    输入参数：
        fs_x: int
            输入的低采样率信号的采样率
        fs_up: int
            目标要取得的增采样后采样率
        x: array_like
            输入的低采样率信号x
        x_len_up: int
            增采样后信号的长度
        up_factor: int
            增采样因子

    返回：
        x_filter: array_like
            通过滤波方法得到的增采样信号 x_filter     
    '''
  
    # 滤波器的技术指标
    fs = fs_up  # 采样频率，等于增采样后的信号采样频率
    f_c = fs_x / 2  # 理想截止频率，等于原信号采样频率的一半
    f_p = f_c - 0.5e3   # 通带截止频率
    f_st = f_c + 0.5e3  # 阻带截止频率
    f_tran = f_st - f_p # 过渡带
    R_p_traget = 1  # 通带最大衰减，in dB
    A_s_traget = 42 # 阻带最小衰减，in dB
    window = 'kaiser'           #采用经分析后效果最好的kaiser窗
    N, kaiser_beta = signal.kaiserord(A_s_traget, f_tran/(0.5*fs))    # 计算凯泽窗口的长度N和β参数
    b = signal.firwin(N, f_c, window=(window, kaiser_beta), fs=fs)
    a = 1

    # Insert zeros between samples.
    x_insert_0 = np.zeros(x_len_up)
    x_insert_0[::up_factor] = x

    # 插零后的信号序列输入数字滤波器进行滤波，并将滤波输出扩大6倍
    x_filter = signal.lfilter(b=b, a=a, x=x_insert_0) * 6

    return x_filter

def  IIR_upsample_filter(fs_x,fs_up,x,x_len_up,up_factor):
    '''
    IIR滤波器增采样处理函数，输入低采样率的信号 x，输出通过滤波方法得到的增采样信号 x_filter。
    输入参数：
        fs_x: int
            输入的低采样率信号的采样率
        fs_up: int
            目标要取得的增采样后采样率
        x: array_like
            输入的低采样率信号x
        x_len_up: int
            增采样后信号的长度
        up_factor: int
            增采样因子

    返回：
        x_filter: array_like
            通过滤波方法得到的增采样信号 x_filter     
    '''
    fs = fs_up          #指标
    f_c = fs_x / 2
    f_p = f_c - 0.5e3
    f_st = f_c + 0.5e3
    R_p_traget = 1  # in dB
    A_s_traget = 42 # in dB
    ftype='cheby2'                      #采用经分析后效果最好的cheby2窗
    b, a = signal.iirdesign(
    f_p, f_st, 
    R_p_traget, A_s_traget,
    ftype=ftype,
    fs=fs)

    # Insert zeros between samples.
    x_insert_0 = np.zeros(x_len_up)
    x_insert_0[::up_factor] = x

    # 插零后的信号序列输入数字滤波器进行滤波，并将滤波输出扩大6倍
    x_filter = signal.lfilter(b=b, a=a, x=x_insert_0) * 6

    return x_filter

def Neighbor_Interpolation(x, up_factor, kind='zero'):
    """
    输入低采样率的信号，输出通过指定插值方法得到的增采样信号
    Args:
        x: array, 待插值信号
        up_factor: int, 上采样因子
        kind: str, 默认是 'zero'
    Returns:
        x_interp: array, 增采样后序列
    """

    x_len = len(x)
    x_sub = np.linspace(0, x_len, x_len)
    x_len_up = np.linspace(0, x_len, 6 * x_len)
    f_interp = interpolate.interp1d(x_sub, x, kind=kind)
    # 所使用的函数为 SciPy 函数库中的 interpolate.interp1d 函数，即为一维插值函数。其中的参数为
    # x_sub: 序列的下标
    # x: 需要增采样的序列
    # kind: 增采样方式
    # 关于kind 候选值
    # 'zero', 'nearest' 阶梯插值，相当于零阶B样条曲线
    # 'slinear', 'linear' 线性插值，用一条直线连接所有的取样点, 相当于一阶B样条曲线
    # 'quadratic', 'cubic' 二阶和三阶B样条曲线，更高阶的曲线可以直接使用整数值指定

    # 该段代码实现了声明一个函数 f_interp，当需要进行增采样时，可以像这样调用
    x_interp = f_interp(x_len_up)
    # 其中，x_len_up 为增采样后的序列的长度

    return x_interp
    
def main(argv):
    # 参数列表：
    #   - 编码指令 PMF文件路径 待编码的输入文件路径 编码后的输出文件路径
    upsample_method = argv[1]
    voice_wav_name = argv[2]
    
    (fs_x, x) = wavfile.read(voice_wav_name)
    up_factor = 6   # upsampling factor
    x_len = len(x)
    fs_up = fs_x * up_factor
    x_len_up = x_len * up_factor
    if upsample_method == 'FIRupsample':
        x_filter = FIR_upsample_filter(fs_x,fs_up,x, x_len_up, up_factor)   # 得到增采样信号 x_filter
    elif upsample_method == 'IIRupsample':
        x_filter = IIR_upsample_filter(fs_x,fs_up,x, x_len_up, up_factor)   # 得到增采样信号 x_filter
    elif upsample_method == 'nearest':          # 最近邻域插值
        x_filter = Neighbor_Interpolation(x,up_factor,kind = 'nearest')     #得到增采样信号x_filter
    elif upsample_method == 'linear':           # 线性插值
        x_filter = Neighbor_Interpolation(x,up_factor,kind = 'linear')      #得到增采样信号x_filter



    output_file_name = argv[3]
    
    wavfile.write(output_file_name, fs_up, np.int16(x_filter))


if __name__ == '__main__':
    main(argv)