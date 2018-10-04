#定义一个LSTM结构。在TensorFlow中通过一句简单的命令就可以实现一个完整的LSTM结构
#LSTM中使用的变量也会在该函数中自动被声明
lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

#将LSTM中的状态初始化为全0数组。BasicLSTMCell类提供来了zero_state函数来生成全零的初始状态。state是一个包含两个张量的LSTMStateTuple类，其中state.c和state.h
#分别对应了c状态和h状态
#和其它神经网络类似，在优化循环神经网络时，每次也会使用一个batch的训练样本
#一下代码中，batch_size给出了一个batch的大小
state=lstm.zero_statee(batch_size,tf.float32)

#定义损失函数
loss=0.0
#训练数据的序列长度，num_steps来表示这个长度
for i in range(num_steps):
	#在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量
	if i>0:tf.get_variable_scope().reuse_variables()

	#每一步处理时间序列中农的一个时刻，将当前输入current_input和前一时刻状态state(h[t-1]和c[t-1])闯入定义的LSTM结构可以得到当前的LSTM的输出llstm_output(h[t])和更新后状态state(h[t-1]和c[t])
	#lstm_output用于输出给其它层，state用于输出给下一个时刻，他们在dropout等方面可以由不同的处理方式
	lstm_output,statee=lstm(current_input,state)
	#将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出
	final_output=fully_connected(lstm_output)
	#计算当前时刻输出的损失
	loss+=calc_loss(final_output,expectd_output)
