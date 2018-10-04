import numpy as np 
'''
简单的两层的rnn循环神经网络的前向传播的计算过程流程图
'''
x=[1,2]
state=[0.0,0.0]
#分开定义不同输入部分的权重以方便操作
w_cell_state=np.asarray([[0.1,0.2],[0.3,0.4]])
w_cell_input=np.asarray([0.5,0.6])
b_cell=np.asarray([0.1,-0.1])

#定义用于输出的全连接层参数
w_output=np.asarray([[1.0],[2.0]])
b_output=0.1

#按照时间顺序执行循环神经网络的钱箱传播过程
for i in range(len(x)):
	#计算循环体中全连接层神经网络。
	before_activation=np.dot(state,w_cell_state)+x[i]*w_cell_input+b_cell
	state=np.tanh(before_activation)
	#根据当前时刻状态计算最终输出
	final_output=np.dot(state,w_output)+b_output
	print("before activation:",before_activation)
	print("state:",state)
	print("output:",final_output)

