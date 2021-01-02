import pygame
from pygame.locals import *
import time
from pyaudio import *
import wave

def play(music):
    # 用文本文件记录wave模块解码每一帧所产生的内容。注意这里不是保存为二进制文件
    mus=music
    if(mus==0):
        return
    if(mus==1):
        print(1)
        dump_buff_file = open(r"ring_01.dup", 'w')
        chunk = 512  # 指定WAV文件的大小
        wf = wave.open(r"ring_01.wav", 'rb')  # 打开WAV文件
    if (mus == 2):
        dump_buff_file = open(r"ring_02.dup", 'w')
        chunk = 512  # 指定WAV文件的大小
        wf = wave.open(r"ring_02.wav", 'rb')  # 打开WAV文件
    if (mus == 3):
        dump_buff_file = open(r"ring_03.dup", 'w')
        chunk = 512  # 指定WAV文件的大小
        wf = wave.open(r"ring_03.wav", 'rb')  # 打开WAV文件
    p = PyAudio()  # 初始化PyAudio模块

    # 打开一个数据流对象，解码而成的帧将直接通过它播放出来，我们就能听到声音啦
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)

    data = wf.readframes(chunk)  # 读取第一帧数据
    #print(data)  # 以文本形式打印出第一帧数据，实际上是转义之后的十六进制字符串

    # 播放音频，并使用while循环继续读取并播放后面的帧数
    # 结束的标志为wave模块读到了空的帧
    while data != b'':
        stream.write(data)  # 将帧写入数据流对象中，以此播放之
        data = wf.readframes(chunk)  # 继续读取后面的帧
        dump_buff_file.write(str(data) + "\n---------------------------------------\n")  # 将读出的帧写入文件中，每一个帧用分割线隔开以便阅读

    stream.stop_stream()  # 停止数据流
    stream.close()  # 关闭数据流
    p.terminate()  # 关闭 PyAudio
    print('play函数结束！')
play(1)