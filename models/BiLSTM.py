import json
import tensorflow as tf
from models.common import random_weight, random_bias, DropoutWrappedLSTMCell, mat_weight_mul

class BiLSTM():
    def __init__(self):
        self.opts = json.load(open("config.json"))["config"]

        # attention V
        self.V=random_bias(self.opts["hidden_size"]*4,name="attention_vector")

    def build(self):
        print("building model...")
        # input
        para = tf.placeholder(dtype=tf.float32, shape=[self.opts["batch"], self.opts["p_length"], self.opts["embedding_dim"]], name="paragraph")
        sen1 = tf.placeholder(dtype=tf.float32,shape=[self.opts["batch"], self.opts["s_length"], self.opts["embedding_dim"]],name="sen1_sentence")
        sen2 = tf.placeholder(dtype=tf.float32,shape=[self.opts["batch"], self.opts["s_length"], self.opts["embedding_dim"]],name="sen2nd_sentence")
        # output
        output = tf.placeholder(dtype=tf.float32, shape=[self.opts["batch"], 2], name="output")


        print("Sentence Encoding")
        # reshape data
        sen1_unstack=tf.unstack(sen1,axis=1) #(b,s,d) => (s,b,d)
        sen2_unstack=tf.unstack(sen2,axis=1) #(b,s,d) => (s,b,d)
        with tf.variable_scope("Sentence_Encoding"):
            fw_cell=[DropoutWrappedLSTMCell(self.opts["hidden_size"],self.opts["in_keep_prob"]) for _ in range(2)]
            bw_cell=[DropoutWrappedLSTMCell(self.opts["hidden_size"],self.opts["in_keep_prob"]) for _ in range(2)]

            print("encoding sentence 1...")
            sen1_full, sen1_fw_final, sen1_bw_final = tf.contrib.rnn.stack_bidirectional_rnn(fw_cell, bw_cell, sen1_unstack,dtype=tf.float32)
            print("encoding sentence 2...")
            sen2_full, sen2_fw_final, sen2_bw_final = tf.contrib.rnn.stack_bidirectional_rnn(fw_cell, bw_cell, sen2_unstack,dtype=tf.float32)

        sen1_enc=tf.stack(sen1_full,axis=1) #(b,s,2h)
        sen2_enc=tf.stack(sen2_full,axis=1) #(b,s,2h)
        print("sen1_enc", sen1_enc)
        print("sen2_enc", sen2_enc)



        print("Output Layer")
        sen1_2 = tf.concat([sen1_enc, sen2_enc], axis=2) # (b,s,4h)
        print("sen1_2:", sen1_2)

        sen1_2_unstack=tf.unstack(sen1_2,axis=1) # (s,b,4h)
        with tf.variable_scope("Output"):
            fw_cell2 = [DropoutWrappedLSTMCell(self.opts["hidden_size"], self.opts["in_keep_prob"]) for _ in range(2)]
            bw_cell2 = [DropoutWrappedLSTMCell(self.opts["hidden_size"], self.opts["in_keep_prob"]) for _ in range(2)]
            lstm_full,lstm_fw_final,lstm_bw_final=tf.contrib.rnn.stack_bidirectional_rnn(fw_cell2,bw_cell2,sen1_2_unstack,dtype=tf.float32)

        lstm_full=tf.stack(lstm_full,axis=1) # (b,s,4h)
        lstm_full=tf.nn.dropout(lstm_full,self.opts['in_keep_prob'])
        print("lstm_full:",lstm_full)

        # # attention layer
        # tanh=tf.tanh(lstm_full)
        # a=tf.nn.softmax(tf.squeeze(mat_weight_mul(tanh,tf.reshape(self.V,shape=[-1,1]))),axis=1) #(b,s,1) =squeeze=>(b,s)
        # print("attention weight:",a)
        # # broadcast
        # tiled_a=tf.concat([tf.reshape(a,shape=[self.opts["batch"],-1,1])]*4*self.opts["hidden_size"],axis=2) # (b,s,4h)
        #
        # lstm_att=tf.multiply(sen1_2,tiled_a) #(b,s,4h)
        # lstm_att=lstm_att[:,self.opts["s_length"],:] # 取最后一层(b,4h)
        # print("lstm_att:",lstm_att)









