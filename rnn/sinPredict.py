import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
'''
对序列值进行预测
对sin函数去离散值，实现对sin函数进行预测的tensorflow程序
'''
HIDDEN_SIZE=30		#LSTM中隐藏节点的个数
NUM_LAYERS=2		#LSTM的层数

TIMESTEPS=10		#循环神经网络的训练序列长度
TRAINING_STEPS=10000	#训练轮数
BATCH_SIZE=32		#batch大小

TRAINING_EXAMPLES=10000		#训练数据的个数
TESTING_EXAMPLES=1000		#测试数据的个数
SAMPLE_GAP=0.01				#采样间隔

def generatee_data(seq):
	x=[]
	y=[]
	#序列的第i项和后面的TIMESTEPS-1项合在一起作为输入：第i+TIMESTEPS项作为输出
	#即用sin函数前面的TIMESTEPS个点的信息预测帝i+TIMESTEPS个点的函数值
	for i in range(len(seq)-TIMESTEPS):
		x.append([seq[i:i+TIMESTEPS]])
		y.append([seq[i+TIMESTEPS]])
	return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(x,y,is_training):
	#使用多层的lstm结构
	cell=tf.nn.rnn_cell.MultiRNNCell([
	tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
	for _ in range(NUM_LAYERS)])

	#使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果
	outputs,_=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
	output=outputs[:,-1,:]
	predictions=tf.contrib.layers.fully_connected(
		output,1,activation_fn=None)
	#只在训练时候计算损失函数和优化步骤，测试时候直接返回预测结果
	if not is_training:
		return predictions,None,None
	#计算损失函数
	loss=tf.losses.mean_squared_error(labels=y,predictions=predictions)

	#创建模型优化器并得到优化步骤
	train_op=tf.contrib.layers.optimize_loss(
		loss,tf.train.get_global_step(),
		optimizer="Adagrad",learning_rate=0.1)
	return predictions,loss,train_op

def train(sess,train_x,train_y):
	#将训练数据以数据集的方式提供给计算图
	ds=tf.data.Dataset.from_tensor_slices((train_x,train_y))
	ds=ds.repeat().shuffle(1000).batch(BATCH_SIZE)
	x,y=ds.make_one_shot_iterator().get_next()

	#调用模型，得到预测结果、损失函数，和训练操作
	with tf.variable_scope("model"):
		predictions,loss,train_op=lstm_model(x,y,True)

	#初始化变量
	sess.run(tf.global_variables_initializer())
	for i in range(TRAINING_STEPS):
		_,l=sess.run([train_op,loss])
		if i%100==0:
			print("train step:"+str(i)+",loss:"+str(l))

def run_eval(sess,test_x,test_y):
	#将预测数据以数据集的方式提供给计算图
	ds=tf.data.Dataset.from_tensor_slices((test_x,test_y))
	ds=ds.batch(1)
	x,y=ds.make_one_shot_iterator().get_next()

	#调用模型得到计算结果，这里不需要输入真是的y值
	with tf.variable_scope("model",reuse=True):
		prediction,_,_=lstm_model(x,[0.0],False)

	#将预测结果存入一个数组
	predictions=[]
	labels=[]
	for i in range(TESTING_EXAMPLES):
		p,l=sess.run([prediction,y])
		predictions.append(p)
		labels.append(l)

	#计算rmse作为评价指标l
	predictions=np.array(predictions).squeeze()
	labels=np.array(labels).squeeze()
	rmse=np.sqrt(((predictions-labels)**2).mean(axis=0))
	print("Mean Square Error is:%f"%rmse)

	#对预测的sin函数曲线进行绘图，得到的结果如下
	plt.figure()
	plt.plot(predictions,label='predictions',linestyle="--",color='red')
	plt.plot(labels,label='real_sin',linestyle="-.",color='black')
	plt.legend()
	plt.show()
	plt.savefig("result.jpg")

if __name__=="__main__":
	test_start=(TRAINING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
	test_end=test_start+(TESTING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
	train_x,train_y=generatee_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES+TIMESTEPS,dtype=np.float32)))
	test_x,test_y=generatee_data(np.sin(np.linspace(
		test_start,test_end,TESTING_EXAMPLES+TIMESTEPS,dtype=np.float32)))
	with tf.Session() as sess:
		#训练模型
		train(sess,train_x,train_y)
		run_eval(sess,test_x,test_y)