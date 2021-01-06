import wave
from sys import argv

def read_wav_data(filename):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和帧速率
    '''
    wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    b = []
    for k in range(num_frame):          #由于默认给出的wav文件为单声道，所以此处单声道读取
        str_data = wav.readframes(1) # 读取全部的帧
    
        c=int.from_bytes(str_data, byteorder='little', signed=True) #因为读取到的帧是以bytes的形式存取的，此处将其转为整型
        b.append(c)

    return b, framerate 

def gen_MSE(records_real, records_predict):
    """
    获得均方差
    Args:
        records_real: array, 真实值
        records_predict: array, 目标值
    Returns:
        float, 均方差
    """
    
    return sum(map(lambda x, y: (x - y)**2, records_real, records_predict)) / len(records_real)


def main(argv):
    # 参数列表：
    #   - 利用程序增采样后的输入文件路径 目标增采样的输入文件路径 程序增采样结果的起始点 目标增采样结果的起始点 比较样本数
    upsample_wav_name = argv[1]
    target_wav_name = argv[2]

    upsample_wav_start = int(argv[3])
    target_wav_start = int(argv[4])
    alain_len = int(argv[5])
    (x1,f1) = read_wav_data(upsample_wav_name)
    (x2,f2) = read_wav_data(target_wav_name)

    if alain_len>len(x1):                       #根据比较样本书，如果不足，进行补0
        for k in range(int(alain_len-len(x1))):
            x1.append(0)
            x2.append(0)
    records_real = x1[upsample_wav_start:upsample_wav_start+alain_len]      #按起始样本点和比较样本数进行截取
    records_predict = x2[target_wav_start:target_wav_start+alain_len]
    MSE = gen_MSE(records_real, records_predict)
    print(MSE)
    


if __name__ == '__main__':
    main(argv)