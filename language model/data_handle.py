import codecs
import sys

#此处input_vacab_file是构件号的词表，通过对raw_data,output_data路径的改变来讲原来的文件映射为数字表示的文本文件
#raw_data,output_data的候选列表为[data/ptb.train.txt,output/ptb.train],[data/ptb.test.txt,output/ptb.test],[data/ptb.valid.txt,output/ptb.valid]
#运行三次分别得到的文件为 output/ptb.train,output/ptb.test,output/ptb.valid
raw_data="data/ptb.valid.txt"#原始的训练集数据文件
input_vocab_file="output/ptb.vocab"#生成的词汇表文件
output_data="output/ptb.valid"#将单词替换成单词编号后的输出文件

#读取词汇表，并建立词汇表到单词编号的映射
with codecs.open(input_vocab_file,"r","utf-8") as f_vocab:
	vocab=[w.strip() for w in f_vocab.readlines()]
word_to_id={k:v for (k,v) in zip(vocab,range(len(vocab)))}

#如果出现了被删除的低频词，则替换为"<unk>"
def get_id(word):
	return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

fin=codecs.open(raw_data,"r","utf-8")
fout=codecs.open(output_data,"w","utf-8")
for line in fin:
	words=line.strip().split()+["<eos>"]#读取单词并添加<eos>结束符
	#将每个单词替换为词汇表中的编号
	out_line=' '.join([str(get_id(w)) for w in words])+"\r\n"
	print(words)
	print(out_line)
	fout.write(out_line)
fin.close()
fout.close()
