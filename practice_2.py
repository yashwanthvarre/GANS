import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
tf.compat.v1.disable_eager_execution()
import utils
import pdb

def plotting(entry, namer):
    # print('a',range(len(entry)), entry)
    plt.plot(range(len(entry)), entry)

    plt.ylabel(namer)
    plt.xlabel('iterations')
    plt.show()
    return
  
train_disc_loss=[]
train_gen_loss=[]



class ConvNet(object):
    def __init__(self):
        print("hello")
        self.batch_size = 50
        #self.keep_prob = tf.constant(0.75)
        self.gstep = tf.compat.v1.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        # print(self.gstep)
        self.skip_step = 500
        self.no_display_imgs = 10
        self.training=False

        self.lr = 0.0003
        self.beta1 = 0.5
        self.gen_train = tf.compat.v1.Variable(True, dtype= tf.bool, trainable = False)
        self.disc_assign = self.gen_train.assign(False)
        # print(self.disc_assign)
        self.gen_assign = self.gen_train.assign(True)
    
    def image_plotter(self, imgs, n = 1, gen = 0, res = 0, n_batches = 0):
        print("hi")
        for i in range(0,self.batch_size, int(self.batch_size/n)):
            
                im = np.reshape(imgs[i,:,:,0], [64,64])
                # print('im',im)
                im = ((im+1)*255.0)/2
                # if gen > 0:
                #     plt.imsave('gen_'+ str(n_batches)+ '_'+ str(i) + '.jpg',im.astype(np.uint8))
                if res > 0:
                    plt.imsave('res_' + str(i) + '.jpg',im.astype(np.uint8), cmap='gray')
                plt.imshow(im.astype(np.uint8), cmap='gray')
                plt.show()



    def get_data(self):
        with tf.name_scope('data'):
            self.true_imgs_dataset = utils.get_celeba_dataset(option =0)
            #self.image_plotter(self.true_imgs_dataset)
            self.true_imgs =  tf.compat.v1.placeholder(dtype = np.float32, shape=[None, 64, 64, 1])
            # print('true',self.true_imgs)



    #generator
    def generator(self):
        z1 = tf.compat.v1.random_normal([self.batch_size, 1, 1,100], mean=0, stddev=0.02)
        # print('z1:',z1)
        filters_start = 512

        with tf.compat.v1.variable_scope("generator"):

            t_conv1 = utils.transpose_conv_bn_relu('t_conv1',z1, filter=int(filters_start), k_size=[4,4], padd="VALID")

            t_conv2 = utils.transpose_conv_bn_relu('t_conv2',t_conv1, filter=int(filters_start/2), k_size=[4,4], stride = [2,2], padd ='SAME')
            # print('conv2:',t_conv2)


            t_conv3 = utils.transpose_conv_bn_relu('t_conv3',t_conv2, filter=int(filters_start/4), k_size=[4,4], stride = [2,2], padd ='SAME')


            t_conv4 = utils.transpose_conv_bn_relu('t_conv4',t_conv3, filter=int(filters_start/8), k_size=[4,4], stride = [2,2], padd ='SAME')

            t_conv5 = tf.compat.v1.layers.conv2d_transpose(t_conv4, 1, [4,4], [2,2], 'SAME', activation=None, name ='t_conv5')
            
            self.fake_imgs = tf.nn.tanh(t_conv5, name='fake_imgs')

            
            
           
    def disc_layers(self,x):
        filters_start = 64
        with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):

            
            #conv1 =utils.conv_bn_leaky_relu('conv1', imgs, filter=int(filters_start), k_size=[4,4], stride=[2,2], padd='SAME')
            
            conv1_without_activation = tf.compat.v1.layers.conv2d(inputs=x,
                                filters=int(filters_start),
                                kernel_size=[4,4],
                                strides=[2,2],
                                padding='SAME', kernel_initializer = tf.random_normal_initializer(0, 0.02), bias_initializer =tf.zeros_initializer(), name='conv1_without_activation')
            conv1 = tf.nn.leaky_relu(conv1_without_activation, name='conv1')
            
            
            
            conv2 =utils.conv_bn_leaky_relu('conv2', conv1, filter=int(filters_start*2), k_size=[4,4], stride=[2,2], padd='SAME')

            conv3 =utils.conv_bn_leaky_relu('conv3', conv2, filter=int(filters_start*4), k_size=[4,4], stride=[2,2], padd='SAME')

            conv4 =utils.conv_bn_leaky_relu('conv4', conv3, filter=int(filters_start*8), k_size=[4,4], stride=[2,2], padd='SAME')

            disc_out = tf.squeeze(tf.compat.v1.layers.conv2d(inputs=conv4,filters=1,kernel_size= [4,4],strides=[1,1],padding='VALID', activation=None), axis=[2,3])
        print("disc_out:",disc_out)
        return disc_out


    #discriminator 
    def discriminator(self): 
        
        labels = tf.cond(self.gen_train, lambda: tf.ones([self.batch_size,1]), lambda: tf.concat([tf.ones([self.batch_size,1]), tf.zeros([self.batch_size,1])], axis=0))
        self.labels= labels   
        
        disc_out_real = self.disc_layers(self.true_imgs)
        # print("disc_out_real:",disc_out_real)
        disc_out_fake = self.disc_layers(self.fake_imgs)
        # print("disc_out_fake:",disc_out_fake)
        
        self.disc_out = tf.cond(self.gen_train, lambda: disc_out_fake, lambda: tf.concat([disc_out_real, disc_out_fake], axis=0))
        
        disc_act =tf.nn.sigmoid(self.disc_out)
        # print('disc_actt',disc_act)
        self.pred = tf.where((disc_act > 0.5), tf.ones_like(self.disc_out), tf.zeros_like(self.disc_out))
        
        self.acc = tf.compat.v1.reduce_mean(tf.cast(tf.equal(labels, self.pred), tf.float32))
        # print('acc_loss',self.acc)

        self.loss = tf.compat.v1.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = self.disc_out))
        # print('loss:',self.loss)
        self.gen_loss = self.loss 
        # print('gen_loss:',self.gen_loss)       
        self.disc_loss = self.loss
        

   
    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        # discriminator AdamOptimizer
        self.disc_opt = tf.compat.v1.train.AdamOptimizer(self.lr, beta1 = self.beta1).minimize(self.disc_loss,
                                                global_step=self.gstep, var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'))
        print('disc_opt',self.disc_opt)
        # generator AdamOptimizer

        self.gen_opt = tf.compat.v1.train.AdamOptimizer(self.lr, beta1 = self.beta1).minimize(self.gen_loss,
                                        global_step=self.gstep, var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
        

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            #tf.summary.scalar('gen_loss', self.gen_loss)
            a=tf.summary.scalar('loss', self.loss)
            print('summary',a)            
            self.summary_op = tf.compat.v1.summary.merge_all()
    

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        print('loaded data')
        self.generator()
        self.discriminator()
        self.optimize()
        self.summary()

        
    def train_one_epoch(self, sess, saver, writer, epoch, step):
        start_time = time.time()
        self.training = True
        n_batches = 0
        total_disc_loss = 0
        total_gen_loss =0
        print("asda ",len(self.true_imgs_dataset))
        for i in range (int( (1*len(self.true_imgs_dataset) )/ self.batch_size)):
            if((i+1) * self.batch_size < len(self.true_imgs_dataset)):
                st = i * self.batch_size
                en = (i+1) * self.batch_size
            else:
                st = i * self.batch_size
                en = len(self.true_imgs_dataset)    
            #print('end:', en)

            ###########################
            ##(1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ##########################
            sess.run(self.disc_assign)
            _, disc_l, acc, disc_pred=  sess.run([self.disc_opt, self.disc_loss, self.acc, self.pred ], {self.true_imgs: self.true_imgs_dataset[st: en]})


            ###########################
            ##(2) Update G network: maximize log(D(G(z)))
            ##########################
            sess.run(self.gen_assign)

            _, gen_l, acc, labels_out, disc_pred, fake_out=  sess.run([self.gen_opt, self.gen_loss, self.acc, self.labels, self.pred, self.fake_imgs], {self.true_imgs: self.true_imgs_dataset[st: en]})

            #writer.add_summary(summaries, global_step=step)
            if (step + 1) % self.skip_step == 0:
                print('step: {0}, Discriminator Loss: {1} generator Loss: {2}'.format(step, disc_l, gen_l))
                self.image_plotter(fake_out, 2, 1, 0, n_batches)               

            step += 1
            total_disc_loss += disc_l
            total_gen_loss += gen_l
            # print("totaldiscloss:",total_disc_loss)
            # print("totalgenloss:",total_gen_loss)

            n_batches += 1
            
        print("asdasd ", n_batches)
        train_disc_loss.append(total_disc_loss/n_batches)   
        train_gen_loss.append(total_gen_loss/n_batches)
        # print("train_disc_loss:",train_disc_loss) 
        # print("train_gen_loss:",train_gen_loss) 
        

        return step


    

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        
        utils.safe_mkdir( '../../checkpoints')
        utils.safe_mkdir('../../checkpoints/convnet_layers')
        writer = tf.compat.v1.summary.FileWriter('../.././graphs/convnet_layers', tf.compat.v1.get_default_graph())
        print('starting to train')
        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
#             saver = tf.compat.v1.train.Saver()
            
#             # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_layers/cifar-convnet1'))
#             # if ckpt and ckpt.model_checkpoint_path:
#             #     saver.restore(sess, ckpt.model_checkpoint_path)
            
            # step = self.gstep.eval()

            # for epoch in range(n_epochs):
            #     step = self.train_one_epoch(sess, saver, writer, epoch, step)
            #     print("epoch ", epoch)
                #self.eval_once(sess, self.test_init, writer, epoch, step)
            
            # new_saver = tf.compat.v1.train.import_meta_graph('my_test_model.meta')
            # new_saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./'))
            print("calling saved model")
            saver.restore(sess, "/home/akarsh/Documents/yash/model/model.ckpt")
            print("restored model")
            fake_out = sess.run(self.fake_imgs)  
            print("Successfully ran model")
            
            self.image_plotter(fake_out, self.no_display_imgs, 0, 1,0)     
            # save_path = saver.save(sess, "/mnt/storage/yash/model/model.ckpt")         
                      
        writer.close()
        


if __name__ == '__main__':
    tf.compat.v1.reset_default_graph()
    model = ConvNet()
    # print(model.batch_size)
    model.build()
    print('built model')    
    #print(model.true_imgs_dataset[0,:,:,0])
    model.train(n_epochs=10)
    # print('done')
   
  
