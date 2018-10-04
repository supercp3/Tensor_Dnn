import tensorflow as tf 

word_label=tf.constant([2,0])
#假设模型对两个单词预测时，产生的logit分别是
#[2.0,-1.0,3.0],[1.0,0.0,-0.5]
predict_logits=tf.constant([[2.0,-1.0,3.0],[1.0,0.0,-0.5]])
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_label,logits=predict_logits)
sess=tf.Session()
sess.run(loss)

word_prob_distribution=tf.constant([[0.0,0.0,1.0],[1.0,0.0,0.0]])
loss=tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution,logits=predict_logits)
sess.run(loss)
