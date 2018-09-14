import json
import math

import tensorflow as tf


class Center_Net():

    def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def random_bias(self, dim, name=None):
        return tf.Variable(tf.truncated_normal([dim]), name=name)

    def random_scalar(self, name=None):
        return tf.Variable(0.0, name=name)

    def DropoutWrappedLSTMCell(self, hidden_size, in_keep_prob, name=None):
        # cell = tf.contrib.rnn.GRUCell(hidden_size)
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    def mat_weight_mul(self, mat, weight):
        # [batch_size, n, m] * [m, p] = [batch_size, n, p]
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])

    def __init__(self):
        self.opts = json.load(open("config.json"))["config"]

        # Weight
        self.W_p_enc=self.random_weight(dim_in=2*self.opts["hidden_size"],dim_out=self.opts["hidden_size"],name="para_enc_weight")
        self.W_1_enc=self.random_weight(dim_in=2 * self.opts["hidden_size"], dim_out=self.opts["hidden_size"], name="sen1_enc_weight")
        self.W_2_enc=self.random_weight(dim_in=2 * self.opts["hidden_size"], dim_out=self.opts["hidden_size"], name="sen2nd_enc_weight")

        self.W_1_v_t_enc = self.random_weight(dim_in=self.opts["hidden_size"], dim_out=self.opts["hidden_size"], name="sen1_enc_last_step_weight")
        self.W_2_v_t_enc = self.random_weight(dim_in=self.opts["hidden_size"], dim_out=self.opts["hidden_size"], name="sen2_enc_last_step_weight")

        self.W_1_gate=self.random_weight(dim_in=4*self.opts["hidden_size"], dim_out=4*self.opts["hidden_size"], name="sen1_gate_weight")
        self.W_2_gate=self.random_weight(dim_in=4*self.opts["hidden_size"], dim_out=4*self.opts["hidden_size"], name="sen2_gate_weight")

        # Attention V
        self.V_s_p_1=self.random_bias(dim=self.opts["hidden_size"], name="sen_para_attention_sen1_v")
        self.V_s_p_2=self.random_bias(dim=self.opts["hidden_size"], name="sen_para_attention_sen2_v")


        # cells
        with tf.variable_scope('SP_match_1') as scope:
            self.SPmatch_cell_1 = self.DropoutWrappedLSTMCell(self.opts['hidden_size'], self.opts['in_keep_prob'])
            self.SPmatch_state_1 = self.SPmatch_cell_1.zero_state(self.opts['batch'], dtype=tf.float32) # the initial state of the rnn
        with tf.variable_scope('SP_match_2') as scope:
            self.SPmatch_cell_2 = self.DropoutWrappedLSTMCell(self.opts['hidden_size'], self.opts['in_keep_prob'])
            self.SPmatch_state_2 = self.SPmatch_cell_1.zero_state(self.opts['batch'],dtype=tf.float32)


    def build(self):
        print("building model...")
        # input
        para=tf.placeholder(dtype=tf.float32,shape=[self.opts["batch"],self.opts["p_length"],self.opts["embedding_dim"]],name="paragraph")
        sen1=tf.placeholder(dtype=tf.float32,shape=[self.opts["batch"],self.opts["s_length"],self.opts["embedding_dim"]],name="sen1_sentence")
        sen2=tf.placeholder(dtype=tf.float32,shape=[self.opts["batch"],self.opts["s_length"],self.opts["embedding_dim"]],name="sen2nd_sentence")
        # output
        output=tf.placeholder(dtype=tf.float32,shape=[self.opts["batch"],2],name="output")


        para_unstack=tf.unstack(value=para,axis=1) # (p,b,d)
        sen1_unstack=tf.unstack(value=sen1,axis=1) # (s,b,d)
        sen2_unstack=tf.unstack(value=sen2,axis=1) # (s,b,d)

        print("Text Encoding Layer")
        with tf.variable_scope("Encoding"):
            fw_cell=[self.DropoutWrappedLSTMCell(hidden_size=self.opts["hidden_size"],in_keep_prob=self.opts["in_keep_prob"]) for _ in range(2)]
            bw_cell=[self.DropoutWrappedLSTMCell(hidden_size=self.opts["hidden_size"],in_keep_prob=self.opts["in_keep_prob"]) for _ in range(2)]

            print("encoding...")
            print("encoding paragraph...")
            para_full,para_fw_final,para_bw_final=tf.contrib.rnn.stack_bidirectional_rnn(fw_cell,bw_cell,para_unstack,dtype=tf.float32)
            print("encoding sentence 1...")
            sen1_full,sen1_fw_final,sen1_bw_final=tf.contrib.rnn.stack_bidirectional_rnn(fw_cell,bw_cell,sen1_unstack,dtype=tf.float32)
            print("encoding sentence 2...")
            sen2_full,sen2_fw_final,sen2_bw_final=tf.contrib.rnn.stack_bidirectional_rnn(fw_cell,bw_cell,sen2_unstack,dtype=tf.float32)

            para_enc=tf.stack(para_full,axis=1) # (b,p,2h)
            sen1_enc=tf.stack(sen1_full,axis=1) # (b,s,2h)
            sen2_enc=tf.stack(sen2_full,axis=1) # (b,s,2h)

            print("para_enc",para_enc)
            print("sen1_enc",sen1_enc)
            print("sen2_enc",sen2_enc)

        print("Sentence-Paragraph Matching")
        sen1_V=[]
        sen2_V=[]
        for t in range(self.opts["s_length"]): # each word of a sentence incorporates information from the whole paragraph
            # reshape
            sen1_t_tiled=tf.concat([tf.reshape(sen1_enc[:,t,:],shape=[self.opts["batch"],1,-1])]*self.opts["p_length"],axis=1) # (b,p,2h)
            sen2_t_tiled=tf.concat([tf.reshape(sen2_enc[:,t,:],shape=[self.opts["batch"],1,-1])]*self.opts["p_length"],axis=1) # (b,p,2h)
            #matmul
            para_W=self.mat_weight_mul(para_enc,self.W_p_enc) # (b,p,h)
            sen1_W=self.mat_weight_mul(sen1_t_tiled,self.W_1_enc) # (b,p,h)
            sen2_W=self.mat_weight_mul(sen2_t_tiled,self.W_2_enc) # (b,p,h)

            if t == 0:
                sen1_tanh=tf.tanh(para_W+sen1_W)
                sen2_tanh=tf.tanh(para_W+sen2_W) # (b,p,h)
            else:
                # reshape v_t contain the information from the previous time step
                sen1_v_t_tiled=tf.concat([tf.reshape(sen1_V[t-1],shape=(self.opts["batch"],1,-1))]*self.opts["p_length"],axis=1) # (b,p,h)
                sen2_v_t_tiled=tf.concat([tf.reshape(sen2_V[t-1],shape=(self.opts["batch"],1,-1))]*self.opts["p_length"],axis=1) # (b,p,h)

                sen1_v_t_W=self.mat_weight_mul(sen1_v_t_tiled, self.W_1_v_t_enc)
                sen2_v_t_W=self.mat_weight_mul(sen2_v_t_tiled, self.W_2_v_t_enc)

                # incorporating sentence paragraph & information form last step's output
                sen1_tanh=tf.tanh(sen1_W+para_W+sen1_v_t_W)
                sen2_tanh=tf.tanh(sen2_W+para_W+sen2_v_t_W)

            sen1_s_t=tf.squeeze(self.mat_weight_mul(sen1_tanh,tf.reshape(self.V_s_p_1,shape=[-1,1]))) # (b,p,1) =squeeze=> (b,p)
            sen2_s_t=tf.squeeze(self.mat_weight_mul(sen2_tanh,tf.reshape(self.V_s_p_1,shape=[-1,1]))) # (b,p,1) =squeeze=> (b,p)

            # attention weight tiled
            sen1_a_t_tiled=tf.concat([tf.reshape(tf.nn.softmax(sen1_s_t,axis=1),shape=[self.opts["batch"],-1,1])]*2*self.opts["hidden_size"],axis=2)
            sen2_a_t_tiled=tf.concat([tf.reshape(tf.nn.softmax(sen2_s_t,axis=1),shape=[self.opts["batch"],-1,1])]*2*self.opts["hidden_size"],axis=2) #(b,p,2h)

            # apply attention
            sen1_c_t=tf.reduce_sum(tf.multiply(para_enc,sen1_a_t_tiled),axis=1)
            sen2_c_t=tf.reduce_sum(tf.multiply(para_enc,sen2_a_t_tiled),axis=1) #(b,2h)

            # adding gate
            sen1_enc_c_t=tf.concat([tf.squeeze(sen1_enc[:,t,:]),sen1_c_t],axis=1) # sen1_enc is additional input in Match-LSTM (b,4h)
            sen2_enc_c_t=tf.concat([tf.squeeze(sen2_enc[:,t,:]),sen2_c_t],axis=1) #(b,4h)

            sen1_enc_c_t=tf.expand_dims(sen1_enc_c_t,axis=1) #(b,1,4h)
            sen2_enc_c_t=tf.expand_dims(sen2_enc_c_t,axis=1) #(b,1,4h)

            # gate
            sen1_enc_c_t_gate=tf.sigmoid(self.mat_weight_mul(sen1_enc_c_t,self.W_1_gate)) # (b,1,4)
            sen2_enc_c_t_gate=tf.sigmoid(self.mat_weight_mul(sen2_enc_c_t,self.W_2_gate)) # (b,1,4)

            # apply gate
            sen1_matched=tf.squeeze(tf.multiply(sen1_enc_c_t,sen1_enc_c_t_gate)) # (b,4h)
            sen2_matched=tf.squeeze(tf.multiply(sen2_enc_c_t,sen2_enc_c_t_gate)) # (b,4h)

            # RNN process
            sen1_v,sen1_state=self.SPmatch_cell_1(sen1_matched,self.SPmatch_state_1) #(b,h)
            sen2_v,sen2_state=self.SPmatch_cell_2(sen1_matched,self.SPmatch_state_2)
            sen1_V.append(sen1_v)
            sen2_V.append(sen2_v)


        sen1_V = tf.stack(sen1_V, 1) # (s,b,h) => (b,s,h)
        sen1_V = tf.nn.dropout(sen1_V, self.opts['in_keep_prob'])
        print('sen1_V:', sen1_V)

        sen2_V = tf.stack(sen2_V, 1) # (s,b,h) => (b,s,h)
        sen2_V = tf.nn.dropout(sen2_V, self.opts['in_keep_prob'])
        print('sen2_V', sen2_V)

        print("Sen1-Sen2 Concatenating")
        sen_con=tf.concat([sen1_V,sen2_V],axis=2)
        print("sen_con:",sen_con)







