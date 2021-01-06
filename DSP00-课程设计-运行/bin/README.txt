# `bin`目录说明

本目录存放课程设计的所有可执行程序及相关文件。

(以下仅为示范)

## 主要子目录和文件说明

- `data/`
  - 存放测试用的数据
- `upsample.exe`
  - 提供了四种方法的增采样程序
- `MSE.exe`
  - MSE计算模块
- `upsample.py`
  - 提供了四种方法的增采样程序的源文件
- `MSE.py`
  - MSE计算模块的源文件

## 运行说明

所有可执行程序都在命令行中运行，没有图形界面。

### upsample.exe

- 运行范例
  `upsample.exe method input output`
  参数解释：
  method 增采样方法 
	  可选FIRupsample、IIRupsample、nearest（最近邻域）、linear（线性）
  input 输入采样率为8kHz的音频信号
  output 输出增采样为48kHz的音频信号

### MSE.exe

- 运行范例
  `MSE.exe upsample_wav_name target_wav_name upsample_wav_start target_wav_start alain_len`
  参数解释：
  upsample_wav_name 利用程序增采样后的输入文件路径
  target_wav_name 目标增采样的输入文件路径
  upsample_wav_start 程序增采样结果的起始点
  target_wav_start 目标增采样结果的起始点
  alain_len 比较样本数



