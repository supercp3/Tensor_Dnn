import numpy as np 
import tensorflow as tf 

train_data="output/ptb.train"#使用单词编号表示的训练数据
train_batch_size=20
train_num_step=35

#从文件中读取数据，并返回包含单词编号的数组
def read_data(file_path):
	with open(file_path,"r") as fin:
		#将整个文档读到一个长字符串
		id_string=' '.join([line.strip() for line in fin.readlines()])
	id_list=[int(w) for w in id_string.split()]#将读取到的单词编号转为整数
	return id_list

#进行数据的切分，首先将所有的训练数据储存到一个list里面，然后将他们映射成为[batch_size]个[num_batches*num_step]个list
#然后对每个[num_batches*num_step]进行切分得到[num_batches]个list
def make_batches(id_list,batch_size,num_step):
	#计算总的batch数量，每个batch包含的单词数量是batch_size*num_step
	num_batches=(len(id_list)-1)//(batch_size*num_step)
	#将数据整理成一个维度为[batch_size,num_batches*num_step]的二维数组
	data=np.array(id_list[:num_batches*batch_size*num_step])
	#data是将所有的数据切分成大小为[batch_size,num_batches*num_step的二维向量,一共batch_size 20份
	#[20,46445]
	data=np.reshape(data,[batch_size,num_batches*num_step])
	print(batch_size,num_batches,num_step)
	print(data.shape)
	#沿着第二个维度将数据切分成num_batches个batch，存入一个数组
	#data_batches是将每一个batch_size的数据再切分为num_batches份，每个batch_size份训练35step
	#1327*20*35
	data_batches=np.split(data,num_batches,axis=1)
	print(data_batches[0])

	#重复上述操作，但是每个位置向右移动一位，这里得到的是RNN每一步输出所需要预测的下一个单词
	label=np.array(id_list[1:num_batches*batch_size*num_step+1])
	label=np.reshape(label,[batch_size,num_batches*num_step])
	label_batches=np.split(label,num_batches,axis=1)
	#返回一个长度为num_batches的数组，其中每一项包括一个data矩阵和一个label矩阵
	print(list(zip(data_batches,label_batches)))
	return list(zip(data_batches,label_batches))


def main():
	train_batches=make_batches(read_data(train_data),train_batch_size,train_num_step)
	#下面是进行模型训练的代码
	
if __name__=="__main__":
		main()