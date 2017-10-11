HW4 - Transfer Learning for Action Recognition
===================

The goal is to recognize various actions being performed in short video clips.  You can read about the UCF-101 dataset here: http://crcv.ucf.edu/data/UCF101.php. There are ~13,000 videos of 101 various actions being performed.

The first part of the assignment will consist of fine-tuning a 50-layer ResNet model which is already pre-trained on ImageNet. I borrowed the code and base model from here: https://github.com/ry/tensorflow-resnet. The original ResNet model has an output layer for the 1000 classes in ImageNet. I've already modified the model for you such that it has 101 outputs for the UCF-101 dataset.

For my own research, I've been trying out a partial ResNet model with some of the convolutional layers being replaced with ConvLSTMs on the Kinetics dataset (released July 2017). This dataset is MUCH larger than UCF-101 (~300K videos of 400 actions): https://deepmind.com/research/open-source/open-source-datasets/kinetics/. After training my model for some time on the Kinetics dataset, I've extracted a vector of length 1024 from the top layer for every frame in the UCF-101 dataset. For the second part of the assignment, you will use these extracted features to train a simple Recurrent Neural Network.

Copy the directory /projects/eot/balq/HW5_Action_Recognition into your own local directory. This contains the pre-trained model, the features I mentioned above, the UCF-101 text files with the train/test splits, and some starter code I will have you finish. The actual UCF-101 data set is located in /projects/eot/balq/UCF101. Last year, I had a huge problem reading videos on BlueWaters. It was extremely slow and frequently had read errors. This year I saved the entire dataset as .hdf5 files. This is poor for space but will read into memory a lot easier. The video files are around 9GB or so while the .hdf5 files are 550+GB. Do not copy this into your own directory. The code is set up to read it from its current location and everyone will be using the same files.

Part 1 - Fine-tuning Resnet for Action Recognition
-------------
The professor plans on going over ResNet in class eventually but you can read about it here: https://arxiv.org/pdf/1512.03385.pdf. As I mentioned in lecture, the UCF-101 dataset is not really big enough to train deep networks on from scratch. Nearly all of the papers I mentioned show around 50%-55% test accuracy regardless of the architecture as it's very easy to overfit and there's only so much you can do with regularization/data augmentation. However, we can take advantage of networks already trained on ImageNet (1mil+ images for 1000 classes). We can pass a single frame from a video clip through a pre-trained ResNet model and end up with a high level "encoding" of what's in the particular frame.

Although videos are a sequence of images, a single image can still classify certain actions very well. You can probably differentiate between "playing the piano" or "bowling" even without watching the entire video segment. For this part of the assignment, we will not leverage any of the temporal information; we will only fine-tune a network to classify actions from a single image.

### $helperFunctions.py$
These are some basic functions which will be necessary later. There's a function for loading in a single frame and performing data augmentation (random crop, flip, adjusting brightness, shrinking, and rotating), a function for determining all of the files belonging to the train/test split as well as their class label, and a function for reading in a sequence of features. I'd recommend checking these functions out. I'll mention them again later when they get used.

### $config.py$
This was part of the original code I borrowed. Truthfully I didn't want to figure it all out and rewrite the model definition so I just left it in. It's essentially used for defining ResNet in TF which I think looks way more complicated than it actually is.

### $single\_frame\_train.py$
This is where you will start adding code. The first 222 lines are given. The $inference()$ function is where you should very easily be able to see the basic operations taking an image as an input and getting the $logits$ as an output. This function calls $stack()$ and $block()$ which are the basic parts making up ResNet. I'll assume you will have a better idea of this after the Professor discusses it in lecture and you check out some of the diagrams in the paper. Let's start adding code.

```python
IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 32
with tf.variable_scope('placeholder'):
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(name='label',dtype=tf.float32,shape=[None,NUM_CLASSES])
    is_training = tf.placeholder(tf.bool ,shape=())
    keep_prob = tf.placeholder(tf.float32 ,shape=())

logits, pred, features = inference(X,
                         num_classes=101,
                         is_training=is_training,
                         bottleneck=True,
                         num_blocks=[3, 4, 6, 3],
                         use_bias=False,
                         keep_prob=keep_prob)
```
The $num\_blocks$ is set up to create a 50-layer ResNet and $num\_classes$ is modified to be 101 as opposed to the original 1000. It's important you set it up exactly this way since this is the model I already have saved for you. 

```python
with tf.variable_scope('loss'):

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                logits = logits, labels = y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
 
    regularization_losses = tf.get_collection(
                            tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = tf.add_n([cross_entropy_mean] + regularization_losses)

with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tvar = tf.trainable_variables()
all_vars = tf.global_variables()

with tf.name_scope('train'):

    opt = tf.train.MomentumOptimizer(0.001, 0.9)
    grads = opt.compute_gradients(loss)

    apply_gradient_op = opt.apply_gradients(grads)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(all_vars)
saver.restore(sess,tf.train.latest_checkpoint('basemodel_UCF101/'))
```
There is nothing out of the ordinary here. If you are unfamiliar with setting up regularization and batch normalization, you can check the functions above for the model definition to see where they pop up and how the these statements modifying the $loss$ and the $train\_op$ tie in. The base model I give you consists of the entirety of a pre-trained ResNet model besides the very last fully connected layer which has been initialized to random values.

```python
data_directory = '/projects/eot/balq/UCF101/'
class_list, train, test = getUCF101(base_directory = data_directory)
```
This is using one of the functions from helperFunctions.py. The $data\_directory$ is set to where all of the .hdf5 files of the video frames are located for everyone. Both $train$ and $test$ are lists of length $2$. The first element of each list is a numpy array of file locations for a train/test sample and the second element is a numpy array of class labels for each particular train/test sample. Here is an example filename: '/projects/eot/balq/UCF101/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi'

I'll use this organizational structure for a lot of things (saved predictions, the extracted features I give you, and the .hdf5 files). For example, the following code is used to read a random video frame (you don't need to put this code in anywhere yourself, it's already located in the $loadImage()$ function of helperFunctions.py).

```python
 filename = filename.replace('.avi','.hdf5')
 filename = filename.replace('UCF-101','UCF-101-hdf5')
 h = h5py.File(filename,'r')
 nFrames = len(h['video'])
 frame_index = np.random.randint(nFrames)
 frame = h['video'][frame_index]
```
The actual location of the video is: '/projects/eot/balq/UCF101/UCF-101-hdf5/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.hdf5'

Now back to adding code to single\_frame\_train.py.

``` python
pool_threads = Pool(10,maxtasksperchild=200)
```
I didn't set up a data reader for this assignment that loads data on the CPU while the GPU is running like we've used for CIFAR10. I think this would require writing a lot of C++ code since Tensorflow doesn't have any built in functions for reading videos (let me know if anyone has something like this set up or can point me in the right direction to do it myself). Since we will have to switch back and forth from reading data and running the network, things can slow down quite a bit (especially since this data is a lot bigger than the CIFAR10 data was). This $pool\_threads$ object allows you to call a function with a list of variables (filenames in our case) which will split up the function calls onto multiple threads before returning a list of all of the function outputs (frames from a video). This is not ideal but still gives a significant performance boost.

```python
for epoch in range(0,20):

    ##### TEST
    test_accuracy = 0.0
    accuracy_count = 0
    random_indices = np.random.permutation(len(test[0]))
    t1 = time.time()
    for i in range(0,500-batch_size,batch_size):
        augment = False
        video_list = [(test[0][k],augment) 
                        for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadFrame,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this batch
                next_batch = 1
        if(next_batch==1):
            continue

        data = np.asarray(data,dtype=np.float32)

        labels_batch = np.asarray(
                        test[1][random_indices[i:(batch_size+i)]]
                        )
        y_batch = np.zeros((batch_size,NUM_CLASSES),dtype=np.float32)
        y_batch[np.arange(batch_size),labels_batch] = 1

        acc_p = sess.run(accuracy,
                feed_dict={X: data, y: y_batch, is_training:False, keep_prob:1.0})

        test_accuracy += acc_p
        accuracy_count += 1
    test_accuracy = test_accuracy/accuracy_count
    print('t:%f TEST:%f' % (float(time.time()-t1), test_accuracy))

    ###### TRAIN
    random_indices = np.random.permutation(len(train[0]))
    for i in range(0, len(train[0])-batch_size,batch_size):

        t1 = time.time()

        augment = True
        video_list = [(train[0][k],augment)
                       for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadFrame,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this
                next_batch = 1
        if(next_batch==1):
            continue

        data = np.asarray(data,dtype=np.float32)

        t_data_load = time.time()-t1

        labels_batch = np.asarray(train[1][random_indices[i:(batch_size+i)]])
        y_batch = np.zeros((batch_size,NUM_CLASSES),dtype=np.float32)
        y_batch[np.arange(batch_size),labels_batch] = 1

        _, loss_p, acc_p = sess.run([train_op,loss,accuracy],
                feed_dict={X: data, y: y_batch, is_training:True, keep_prob:0.5})

        t_train = time.time() - t1 - t_data_load

        print('epoch: %d i: %d t_load: %f t_train: %f loss: %f acc: %f'
                     % (epoch,i,t_data_load,t_train,loss_p,acc_p))
```
This is a lot of code but it's simply set up just like the other basic training loops we've seen. There is both a test and train section (you can technically comment out the test part if you're running it on BlueWaters and have no need to see the progress as it trains). The $pool\_threads.map()$ function call is what I mentioned previously about splitting the data reading such that it uses multiple CPU cores. The code is set up to run for $20$ epochs. Even though we're only seeing $20$ random frames from the videos, this is enough to saturate learning and achieve nearly $100\%$ on the training data. The test output will most likely be floating around $73\%$ to $77\%$ but this is for only a single frame. We can then take the mean output for the entire video for our actual classification.

```python
if not os.path.exists('single_frame_model/'):
    os.makedirs('single_frame_model/')
all_vars = tf.global_variables()
saver = tf.train.Saver(all_vars)
saver.save(sess,'single_frame_model/model')

pool_threads.close()
pool_threads.terminate()
```
Save the model and close the multiprocessing pool object.

Some of the videos are $400+$ frames and can take a long time to evaluate. Let's save the prediction for every video so we can compare it to the results from part 2.

```python
prediction_directory = 'UCF-101-predictions/'
if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)
for label in class_list:
    if not os.path.exists(prediction_directory+label+'/'):
        os.makedirs(prediction_directory+label+'/')
```
This creates an identical directory structure to how the videos are stored except for 'UCF-101-predictions/' being the base directory.

```python

acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
for i in range(len(test[0])):

    t1 = time.time()

    index = random_indices[i]

    filename = test[0][index]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-hdf5')

    h = h5py.File(filename,'r')
    nFrames = len(h['video'])

    data = np.zeros((nFrames,IMAGE_SIZE,IMAGE_SIZE,3),dtype=np.float32)
    prediction = np.zeros((nFrames,NUM_CLASSES),dtype=np.float32)

    for j in range(nFrames):
        frame = h['video'][j] - mean_subtract
        frame = cv2.resize(frame,(IMAGE_SIZE,IMAGE_SIZE))
        frame = frame.astype(np.float32)
        data[j,:,:,:] = frame
    h.close()

    loop_i = list(range(0,nFrames,400))
    loop_i.append(nFrames)

    for j in range(len(loop_i)-1):
        data_batch = data[loop_i[j]:loop_i[j+1]]

        curr_pred = sess.run(pred,
            feed_dict={X: data_batch, is_training:False, keep_prob:1.0})
        prediction[loop_i[j]:loop_i[j+1]] = curr_pred

    filename = filename.replace(data_directory+'UCF-101-hd5/',prediction_directory)
    if(not os.path.isfile(filename)):
        with h5py.File(filename,'w') as h:
            h.create_dataset('predictions',data=prediction)

    label = test[1][index]
    prediction = np.sum(np.log(prediction),axis=0)
    argsort_pred = np.argsort(-prediction)[0:10]

    confusion_matrix[label,argsort_pred[0]] += 1
    if(label==argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5]==label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:]==label)):
        acc_top10 += 1.0

    print('i:%d nFrames:%d t:%f (%f,%f,%f)' 
          % (i,nFrames,time.time()-t1,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))

number_of_examples = np.sum(confusion_matrix,axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])
```
This loops through the dataset one video at a time saving the results. If the video is less than $400$ frames (limited by GPU memory), all of the frames are processed at once and stored in $prediction$. If it's more than $400$ frames, it loops through batches of $400$ until the entire video is processed. The rest of the code reports top-1, top-5, and top-10 accuracy as well as creates a confusion matrix. I won't go into detail here since I will ask you to do this later.

At this point, the model should be getting around $79\%$ to $81\%$ (full video prediction).

Part 2 - Extracted Features from ConvLSTM
-------------
As mentioned in the introduction, I've provided features of dimension $1024$ for every frame in UCF-101 which are the last layer features of a convolutional LSTM based model trained on the Kinetics dataset. You'll now create a new script to train a recurrent neural network on these features.

### $train\_rnn.py$

```python
import tensorflow as tf

import numpy as np
import os
import time

from multiprocessing import Pool

from helperFunctions import getUCF101
from helperFunctions import loadSequence
import h5py
```
```python
data_directory = ''
class_list, train, test = getUCF101(base_directory = data_directory)
```
The $data\_directory$ is just your base directory this time (hence the empty single quotes). This is where 'UCF-101-features' should be located.

```python
sequence_length = 15
sequence_length_test = 200
batch_size = 64
num_of_features = 1024
num_classes = 101

W_output = tf.get_variable('W_output', 
                            [num_of_features, num_classes], 
                            initializer=tf.contrib.layers.xavier_initializer())
B_output = tf.get_variable('B_output', 
                            [num_classes], 
                            initializer=tf.constant_initializer())

X_sequence = tf.placeholder(tf.float32, 
                            [sequence_length, batch_size, num_of_features])
X_sequence_test = tf.placeholder(tf.float32,
                            [sequence_length_test, batch_size, num_of_features])
y = tf.placeholder(name='label',dtype=tf.float32,shape=[batch_size,num_classes])
keep_prob = tf.placeholder(tf.float32 ,shape=())

lstm = tf.contrib.rnn.BasicLSTMCell(num_of_features)
hidden_state = tf.zeros([batch_size, num_of_features])
current_state = tf.zeros([batch_size, num_of_features])
state = hidden_state, current_state
state_test = hidden_state, current_state
```
For the variables $X\_sequence$ and $X\_sequence\_test$, notice there is an extra dimension compared to a normal input. We set up the placeholder to contain a full sequence of frames as opposed to a single frame. We will train the model on short sequences and test the model on longer sequences is the reason for the distinction between these two placeholders.

The $BasicLSTMCell$ with $n$ number of features requires a $hidden\_state$ and $current\_state$ of size $bs\times n$. These are used to initalize the $state$ and $state\_test$ variables each time it sees a new sequence (instead of starting with the states always equal to $0$, $hidden\_state$ and $current\_state$ could be replaced with variable definitions such that the starting state is then a trainable parameter if desired).

```python
probabilities = []
loss = 0.0
for i in range(sequence_length):
    output, state = lstm(X_sequence[i,:,:], state)
    output = tf.nn.dropout(output,keep_prob=keep_prob)
    logits = tf.matmul(output, W_output) + B_output

    probabilities.append(tf.nn.softmax(logits))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits = logits, labels = y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss += cross_entropy_mean/float(sequence_length)

probabilities_test = []
for i in range(sequence_length_test):
    output_test, state_test = lstm(X_sequence_test[i,:,:], state_test)
    output_test = tf.nn.dropout(output_test,keep_prob=keep_prob)
    logits_test = tf.matmul(output_test, W_output) + B_output

    probabilities_test.append(tf.nn.softmax(logits_test))
```
These loops are set up to process the full sequence. Notice the loss is continuously being added to. We are making a prediction after every frame and they all contribute to the total loss.

```python

tvar = tf.trainable_variables()
all_vars = tf.global_variables()

opt = tf.train.MomentumOptimizer(0.01, 0.9)
grads = opt.compute_gradients(loss)
train_op = opt.apply_gradients(grads)

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(all_vars)

pool_threads = Pool(10,maxtasksperchild=200)
```

```python
for epoch in range(0,100):

    # ###### TEST
    if(epoch%5==0):
        random_indices = np.random.permutation(len(test[0]))
        prob_test = 0.0
        count = 0
        is_train = False
        for i in range(0, 500-batch_size,batch_size):
            video_list = [(test[0][k],sequence_length_test,is_train)
                            for k in random_indices[i:(batch_size+i)]]
            data = pool_threads.map(loadSequence,video_list)

            next_batch = 0
            for video in data:
                if video.size==0: # there was an exception, skip this
                    next_batch = 1
            if(next_batch==1):
                continue

            data = np.asarray(data,dtype=np.float32)
            data = data.transpose(1,0,2)

            labels_batch = np.asarray(test[1][random_indices[i:(batch_size+i)]])
            y_batch = np.zeros((batch_size,num_classes),dtype=np.float32)
            y_batch[np.arange(batch_size),labels_batch] = 1

            prob = sess.run(probabilities_test,
                   feed_dict={X_sequence_test: data, keep_prob:1.0})

            prob = np.asarray(prob)
            log_prob = np.log(prob)
            log_prob = np.mean(log_prob,axis=0)
            pred = np.argmax(log_prob,axis=1)

            prob_p = np.sum(pred==labels_batch)/float(batch_size)

            prob_test += prob_p
            count += 1

        print('TEST: %f' % (prob_test/count))

    ###### TRAIN
    random_indices = np.random.permutation(len(train[0]))
    count = 0
    is_train = True
    for i in range(0, len(train[0])-batch_size,batch_size):

        t1 = time.time()

        video_list = [(train[0][k],sequence_length,is_train)
                     for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadSequence,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this
                next_batch = 1
        if(next_batch==1):
            continue

        data = np.asarray(data,dtype=np.float32)
        data = data.transpose(1,0,2)

        t_data_load = time.time()-t1

        labels_batch = np.asarray(train[1][random_indices[i:(batch_size+i)]])

        y_batch = np.zeros((batch_size,num_classes),dtype=np.float32)
        y_batch[np.arange(batch_size),labels_batch] = 1

        _, loss_p, prob = sess.run([train_op,loss,probabilities],
                          feed_dict={X_sequence: data, y: y_batch, keep_prob:1.0})

        prob = np.asarray(prob)
        log_prob = np.log(prob)
        log_prob = np.mean(log_prob,axis=0)
        pred = np.argmax(log_prob,axis=1)

        prob_p = np.sum(pred==labels_batch)/float(batch_size)
        t_train = time.time() - t1 - t_data_load

        count += 1
        if(count%10 == 0):
            print('epoch: %d i: %d t_load: %f t_train: %f loss: %f acc: %f'
                  % (epoch,i,t_data_load,t_train,loss_p,prob_p))

if not os.path.exists('rnn_model/'):
    os.makedirs('rnn_model/')
all_vars = tf.global_variables()
saver = tf.train.Saver(all_vars)
saver.save(sess,'rnn_model/model')

pool_threads.terminate()
pool_threads.close()
```
Notice the lists $probabilities$ and $probabilities\_test$ we created in the model definition can be pulled out in their entirety. I sum over $log(p)$ and choose the $argmax$ to get my actual sequence classification. This code should run very fast and you can try out training with/without dropout, different sequence lengths, etc. I'd suggest doing this part in interactive mode since you'll be able to get a better idea of how RNNs work in general. This should achieve around $82\%$ on the test dataset.

### Some comments
Truthfully, an RNN is completely unnecessary for this part. My best results (around $84\%$) came from setting sequence length to $1$ and skipping over the LSTM layer.

```python
logits = tf.matmul(X_sequence[i,:,:], W_output) + B_output
```
I was hoping to see a larger improvement to justify using an RNN for the sake of the assignment, but the features don't actually change all that much over time.  Consider the convLSTM network I trained. The input (sequence of images) is constantly changing but the output is static (same label for every frame). Although the features early in the network are changing on a frame by frame basis, by the time you get to the higher level features, they are relatively still considering it's trying to estimate a static label. Even if you only look at features from a single frame, these features were calculated from a sequence of frames. By setting the sequence length to $1$, you're still actually using more information than what is provided from a single frame. Within just a few epochs, you'll most likely max out your training accuracy. 

There's a tradeoff between how we trained the model in part 1 compared to part 2. In part 1, we trained the top layer from scratch and fine-tuned the rest of the model. If the top layer features for the training set are not linearly separable, that's still fine since we can tweak the layers below it. We can also perform data augmentation on the original input frames which helps the model generalize well to the test dataset. 

In part 2, if the features I extracted for the training set were not linearly separable (although as shown above, it was separable with a simple linear classifier), we would've been forced to increase the model capacity by adding more layers as opposed to fine-tuning the layers of the convLSTM. We also lost the ability to perform data augmentation (although if you check $loadSequence()$ in helper\_functions.py, there is slight data augmentation since I repeat the sequence if it's too short and have a random chance to play it in reverse during training). The benefit of how we did it in part 2 is training time. We don't need to evaluate the entire network every single time. The convLSTM model has to use a very small batch size (due to GPU memory limits) and is SIGNIFICANTLY slower than what we did here. 


Comparing the two models
-------------
Save the predictions from part 2 in another directory just like we did in part 1 (either with your RNN model or the single linear output layer, whichever you prefer). I extracted at most $200$ frames worth of features for each video in part 2 as opposed to the whole video. Also, as I mentioned, the $loadSequence()$ function actually repeats the sequence if it's less than the desired sequence length. This should make things simple since you can just set the sequence length to $200$ and not have to worry about extracting the predictions for variable length sequences. You'll need to come up with the code yourself here (although it's very similar to what I did earlier when saving the predictions for part 1).

### Combine .hdf5 predictions 

```python
data_directory = ''
class_list, train, test = getUCF101(base_directory = data_directory)

###### some other code
###### you have in between

filenames = test[0][random_indices[i:(batch_size+i)]]
log_prob_single = np.zeros((batch_size,num_classes),np.float32)

for j in range(len(filenames)):
    filename = filenames[j]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-predictions')
    with h5py.File(filename,'r') as h:
        pred_new = h['predictions'][:]
    log_prob_single[j,:] = np.mean(np.log(pred_new),axis=0)
```
This is an example for reading a random batch of predictions from part 1. The variable $log\_prob\_single$ is size $bs\times 101$ containing the log likelihood for each class averaged over the entire video. Combine this with the prediction from part 2 to see the test accuracy when using the output from both models.

What to Turn In
-------------
(top-1,top-5,top-10) accuracy for the model in part 1, the model in part 2, and their combined outputs. As a sanity check, the top-1 accuracy should be around $79\%$-$81\%$, $82\%$-$84\%$, and  $87\%$-$89\%$ for the three. There should be a noticeable difference when combining the two outputs. Both networks are set up with the capability to learn different features (one purely static, the other static+temporal)

Confusion matrix for the model in part 1, confusion matrix for the model in part 2, and confusion matrix for the combined outputs. A simple way is to save these as images using a function like pcolormesh(matrix) from matplotlib.pyplot.

Calculate the difference in performance for each class between the part 1 model and the combined output model. Report the 10 classes with the largest improvement and the 10 classes with worse performance (or smallest improvement if they all improved). See if you can think of any explanation (I'm not implying there is one, I didn't actually do this exact calculation myself and I'm curious).

Compress all the code, your directories with the predictions, and the items mentioned above. Turn this in.

(I wouldn't do all of this until the very end. It'll be much easier to save all of the predictions as you go along, then do all of this at the end in a new script.)

EXTRA CREDIT OPTIONS:
-----
Just like you saved the predictions in part 1, save the last layer features (they are labeled $x$ in the code I gave you). Then repeat part 2 except modify $loadSequence()$ such that it concatenates the features from part 1 and 2 together. Train an RNN on these combined features.

There is a dataset called BU-101: http://cs-people.bu.edu/sbargal/BU-action/. This contains ~27,000 images scraped from the web that directly match with the 101 labels from UCF-101. Repeat part 1 where you switch off between using images from BU-101 and frames from UCF-101. You should see a $1\%$-$3\%$ increase in the test accuracy. I have the dataset downloaded already and a function for reading images (as opposed to .hdf5 files). Let me know if there is any interest and I can help out.

