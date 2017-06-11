import network as inference
import tensorflow as tf
import load_data
mnist = load_data.read_data_sets(one_hot=True, train_image_number=360000, test_image_number=1000)

IS_POSITION_BASED = True

def pre_processing(images, interpolation_method, size):
    processed_images = []
    for image in images:
        channel_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        intermidiate_resize = cv2.resize(channel_1, size, interpolation= interpolation_method)
        resized = cv2.resize(intermidiate_resize, (28, 28), interpolation = cv2.INTER_AREA)
        unrolled = np.array(resized, dtype="float").flatten()
        normalized = np.divide(unrolled, 255)
        processed_images.append(normalized)
        return processed_images

# Parameters
learn_rate = 0.001
decay_rate = 0.1
batch_size = 128
display_step = 2000

n_classes = 10
dropout = 0.8 # Dropout, probability to keep units
imagesize = 28
img_channel = 1

if IS_POSITION_BASED:
    nx, ny = (imagesize, imagesize)
    xt = np.linspace(0, 1, nx) # use 1/56 to 1-1/56 for line spacing to take intersecting point to the center of image pixel
    yt = np.linspace(0, 1, ny)
    xpos, ypos = np.meshgrid(xt, yt)
    xpos = np.array(xpos).flatten()
    ypos = np.array(ypos).flatten()
    img_channel = 3


x = tf.placeholder(tf.float32, [None, imagesize, imagesize, img_channel])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

pred = inference.alex_net(x, keep_prob, n_classes, imagesize, img_channel)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learn_rate, global_step, 1000, decay_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
saver = tf.train.Saver()
tf.add_to_collection("x", x)
tf.add_to_collection("y", y)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("pred", pred)
tf.add_to_collection("accuracy", accuracy)

with tf.Session() as sess:
    if IS_POSITION_BASED:
        saver.restore(sess, "./logs/model/position/model.ckpt")
    else:
        saver.restore(sess, "./logs/model/normal/model.ckpt")
    print("Model restored.")

    sess.run(init)
    step = 1
    if IS_POSITION_BASED:
        train_writer = tf.summary.FileWriter('./logs/position_train', sess.graph)
    else:
        train_writer = tf.summary.FileWriter('./logs/normal_train', sess.graph)
    while step <= epochs * 60000/ batch_size:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        train_image = pre_processing(batch_x, interpolation_flag = cv2.INTER_CUBIC, size=(28, 28))
        input_batch = train_image
        if IS_POSITION_BASED:
            input_batch =  np.swapaxes(np.swapaxes(np.vstack([[train_image],[[xpos,]*batch_size], [[ypos,]*batch_size]]), 0, 2), 0, 1)
            
        sess.run(optimizer, feed_dict={x: input_batch, y: batch_y, keep_prob: dropout})
        
        if step % display_steps == 0:
            summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={x: input_batch, y: batch_y, keep_prob:1.0})
            train_writer.add_summary(summary, step)

        if step*batch_size % 1000 == 0:
            print "TRAINING IMAGE= {:.1f}".format(step*batch_size)+", EPOCH= {:.5f}".format(step*batch_size//60000)+", loss= {:.6f}".format(loss) + ", Accuracy= {:.5f}".format(acc)

        if step % int(60000/batch_size) == 0:
            validation_images = pre_processing(mnist.validation.images[:validation_examples], interpolation_flag=cv2.INTER_CUBIC, size=(10, 10))
            # validation_images = mnist.test.images[validation_examples:test_examples + validation_examples]
            
            validation_data = validation_images
            if IS_POSITION_BASED:
                validation_data = np.swapaxes(np.swapaxes(np.vstack([[validation_images],[[xpos,]*validation_examples], [[ypos,]*validation_examples]]), 0, 2), 0, 1)                

            loss, vacc = sess.run([cost, accuracy], feed_dict={x: validation_data, y: mnist.validation.labels[:validation_examples], keep_prob:1.0})

            print "VALIDATION LOSS= {:.6f}".format(loss) + ", VALIDATION ACCURACY= {:.5f}".format(vacc)

            if vacc >0.9964:
                break
            elif vacc < 0.9780:
                learning_rate = 0.00001


        if step % 500 == 0:
            if IS_POSITION_BASED:
                save_path = saver.save(sess, "./logs/model/position/model.ckpt")
            else:
                save_path = saver.save(sess, "./logs/model/normal/model.ckpt")
            print "Model saved in file: %s" % save_path

        step += 1
    print "Optimization Finished!"

    if IS_POSITION_BASED:
        save_path = saver.save(sess, "./logs/model/position/model.ckpt")
    else:
        save_path = saver.save(sess, "./logs/model/normal/model.ckpt")
    print "Model saved in file: %s" % save_path
    

    test_images = pre_processing(mnist.test.images[:test_examples], interpolation_flag= cv2.INTER_CUBIC, size=(10, 10))
    # test_images = mnist.test.images[:test_examples]
    test_data = test_images
    if IS_POSITION_BASED:
        test_data = np.swapaxes(np.swapaxes(np.vstack([[test_images],[[xpos,]*test_examples], [[ypos,]*test_examples]]), 0, 2), 0, 1)
    
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: mnist.test.labels[:test_examples], keep_prob:1.0})