import codecs
import collections
from operator import itemgetter
#源文件
raw_data="data/ptb.train.txt"
#经过处理以后的输出词汇表文件
vocab_output="output/ptb.vocab"

counter=collections.Counter()
with codecs.open(raw_data,"r","utf-8") as f:
	for line in f:
		for word in line.strip().split():
			counter[word]+=1

sorted_word_to_cnt=sorted(counter.items(),key=itemgetter(1),reverse=True)
sorted_words=[x[0] for x in sorted_word_to_cnt]
sorted_words=["<eos>"]+sorted_words
#sorted_words=["<unk>","<sos>","<eos>"]+sorted_words
#if len(sorted_words)>10000:
#	sorted_words=sorted_words[:10000]
print(sorted_word_to_cnt[:10])
with codecs.open(vocab_output,"w","utf-8") as file_output:
	for word in sorted_words:
		file_output.write(word+'\r\n')#'\n'换行在txt文件里面不显示换行，使用'\r\n'就可以显示换行了
#确定词汇表vocab_output
#接下来需要将词汇表进行编号，然后将训练数据和测试数据都表示成编号的形式
