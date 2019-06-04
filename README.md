# 股市预测模型

这个模型是[VivekPa:AIAlpha](https://github.com/VivekPa/AIAlpha) 的，我拿过来修改成 Tensorflow 实现，并重新调试。

    python preprocessing.py

是进行小波变换，数据预处理过滤噪声用的。

    python autoencoder.py

是利用编码解码器进行数据维度压缩用的。

    python data_processing.py

用来拆分数据，拆成训练集和测试集。

    python model.py

构造 LSTM 模型，训练并预测数据用。

按照上面几个脚本的顺序执行可以重现实验结果：

![](result.png)
