#IITM/Sem2/DL/DL\ Assign/Assign3/
from math import exp
from random import seed
from random import random
from csv import reader
import argparse
import numpy as np
import os
import pickle
import sys
from sklearn.decomposition import PCA
np.random.seed(1234)
# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#Initialize weights and bias for a given network
def initialize_net():
    W[0] = np.random.randn(sizes[0], num_input)/np.sqrt(num_input)
    b[0] = np.random.randn(sizes[0])/np.sqrt(num_input)
    if opt=='momentum' or opt=='nag':
        tmp_delta_W[0] = np.zeros((sizes[0], num_input))
        tmp_delta_b[0] = np.zeros(sizes[0])
    if opt=='adam':
        m_delta_W[0] = np.zeros((sizes[0], num_input))
        m_delta_b[0] = np.zeros(sizes[0])
        v_delta_W[0] = np.zeros((sizes[0], num_input))
        v_delta_b[0] = np.zeros(sizes[0])
    for i in range(num_hidden-1):
        W[i+1] = np.random.randn(sizes[i+1],sizes[i])/np.sqrt(sizes[i])
        b[i+1] = np.random.randn(sizes[i+1])/np.sqrt(sizes[i])
        if opt=='momentum' or opt=='nag':
            tmp_delta_W[i+1] = np.zeros((sizes[i+1],sizes[i]))
            tmp_delta_b[i+1] = np.zeros(sizes[i+1])
        if opt=='adam':
            m_delta_W[i+1] = np.zeros((sizes[i+1],sizes[i]))
            m_delta_b[i+1] = np.zeros(sizes[i+1])
            v_delta_W[i+1] = np.zeros((sizes[i+1],sizes[i]))
            v_delta_b[i+1] = np.zeros(sizes[i+1])
    W[num_hidden] = np.random.randn(num_output, sizes[num_hidden-1])/np.sqrt(sizes[num_hidden-1])
    b[num_hidden] = np.random.randn(num_output)/np.sqrt(sizes[num_hidden-1])
    if opt=='momentum' or opt=='nag':
        tmp_delta_W[num_hidden] = np.zeros((num_output, sizes[num_hidden-1]))
        tmp_delta_b[num_hidden] = np.zeros(num_output)
    if opt=='adam':
        m_delta_W[num_hidden] = np.zeros((num_output, sizes[num_hidden-1]))
        m_delta_b[num_hidden] = np.zeros(num_output)
        v_delta_W[num_hidden] = np.zeros((num_output, sizes[num_hidden-1]))
        v_delta_b[num_hidden] = np.zeros(num_output)

def initialize_delta():
    delta_W[0] = np.zeros((sizes[0], num_input))
    delta_b[0] = np.zeros(sizes[0])
    for i in range(num_hidden-1):
        delta_W[i+1] = np.zeros((sizes[i+1],sizes[i]))
        delta_b[i+1] = np.zeros(sizes[i+1])
    delta_W[num_hidden] = np.zeros((num_output, sizes[num_hidden-1]))
    delta_b[num_hidden] = np.zeros(num_output)


def normalize_data():
    train_set_mean = train_set.mean(axis = 0, keepdims = True)
    train_set[:, 0:784] = train_set[:, 0:784].astype(np.float64)/np.float64(normalize_factor)
    validation_set[:, 0:784] = validation_set[:, 0:784].astype(np.float64)/np.float64(normalize_factor)
    test_set[:, 0:784] = test_set[:, 0:784].astype(np.float64)/np.float64(normalize_factor)
    train_set_mean = train_set[:,0:784].mean(axis = 0, keepdims = True)
    train_set[:, 0:784]-=train_set_mean
    validation_set[:, 0:784]-=train_set_mean
    test_set[:, 0:784]-=train_set_mean

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def sigmoid(a):
    a = np.clip(a, -600, 600)
    return (1.0 / (1.0 + np.exp(-1.0 * a)))

def leakyRelu(a):
    return np.where(a>0, a, a*0.01)

def forward_propagate(h):
    for i in range(num_hidden):
        a = np.add(np.dot(W[i],h),b[i])
        if(activation == 'tanh'):
            h = np.tanh(a)
        else:
            h = sigmoid(a)
        outputsM[i] = h
        activationM[i] = a
    a = np.add(np.dot(W[num_hidden], h),b[num_hidden])
    activationM[num_hidden] = a
    outputsM[num_hidden] = softmax(a)
    return 0


def calculate_loss(actual_output):
    if loss_type == 'sq':
        loss = np.inner(np.subtract(actual_output, outputsM[num_hidden]), np.subtract(actual_output, outputsM[num_hidden]))
    else:
        loss = -np.amin(np.multiply(actual_output, np.ma.log2(outputsM[num_hidden]).filled(0)))
    return loss

def calculate_total_loss_train():
    total_loss_train = 0
    train_set_len = len(train_set)
    predict_label_train = [None]*(train_set_len)
    for i in range(train_set_len):
        forward_propagate(train_set[i][0:num_input])
        actual_output = np.zeros(10)
        actual_output[int(train_set_op[i])] = 1
        total_loss_train+=calculate_loss(actual_output)
        predict_label_train[i] = np.argmax(outputsM[num_hidden])
    total_error_train = 1 - (np.sum(predict_label_train == (train_set_op.astype(int)))/train_set_len)
    return total_loss_train, total_error_train

def calculate_total_loss_val():
    total_loss_val = 0
    val_set_len = len(validation_set)
    predict_label_val = [None]*(val_set_len)
    for i in range(val_set_len):
        forward_propagate(validation_set[i][0:num_input])
        actual_output = np.zeros(10)
        actual_output[int(validation_set_op[i])] = 1
        total_loss_val+=calculate_loss(actual_output)
        predict_label_val[i] = np.argmax(outputsM[num_hidden])
    total_error_val = 1 - (np.sum(predict_label_val == (validation_set_op.astype(int)))/val_set_len)
    return total_loss_val, total_error_val


def transfer_derivative(activation_cur):
    if activation == 'tanh':
        gdash = 1 - (np.tanh(activation_cur)**2)
    elif activation=='leakyrelu':
        alpha = 0.01
        gdash = np.ones_like(activation_cur)
        gdash[activation_cur<0] = alpha
    else:
        gdash = np.multiply(sigmoid(activation_cur),(np.ones(activation_cur.shape)-sigmoid(activation_cur)))
    
    return gdash

def backward_propagate(actual_output, input_row):
    if loss_type == 'sq':
        diag_op = np.diag(outputsM[num_hidden])
        dldy = 2*np.transpose(outputsM[num_hidden]-actual_output)
        minus_op = np.dot(np.multiply(-1, np.ones((num_output, num_output))),diag_op)
        dyda = np.dot(diag_op, np.add(np.identity(num_output), minus_op))
        dldak = np.dot( dldy,dyda)
    else:
        dldak = (outputsM[num_hidden]-actual_output)
    for i in range(num_hidden, -1, -1):
        if i!=0:
            delta_W[i] += np.outer(dldak, outputsM[i-1])
        else:
            delta_W[i] += np.outer(dldak, np.transpose(input_row))
        delta_b[i] += dldak
        dldhk = np.dot(np.transpose(W[i]), dldak)
        if i==0:
            break
        gdash = transfer_derivative(activationM[i-1])
        dldak =  np.multiply(dldhk,gdash)
    return 0

def update_weights(epoch, batch):
    global step_counter,log_str_arr_indx
    step_counter+=1
    t = epoch*batch_size+1+batch
    if opt=='gd':
        for i in range(num_hidden+1):
            W[i] = W[i] - lr*(delta_W[i])
            b[i] = b[i] - lr*(delta_b[i])
    elif opt=='momentum' or opt=='nag':
        for i in range(num_hidden+1):
            delta_W[i] = momentum * tmp_delta_W[i]+ lr*(delta_W[i])
            delta_b[i] = momentum * tmp_delta_b[i]+ lr*(delta_b[i])
            W[i] = W[i] - (delta_W[i]) 
            b[i] = b[i] - (delta_b[i])
            tmp_delta_W[i] = delta_W[i]
            tmp_delta_b[i] = delta_b[i]
    elif opt=='adam':
        beta1=0.9
        beta2=0.999
        epc = 10**-8
        for i in range(num_hidden+1):
            m_delta_W[i] = beta1*m_delta_W[i] + (1- beta1)*(delta_W[i])
            m_delta_b[i] = beta1*m_delta_b[i] + (1- beta1)*(delta_b[i])
            v_delta_W[i] = beta2*v_delta_W[i] + (1- beta2)*((delta_W[i])**2)
            v_delta_b[i] = beta2*v_delta_b[i] + (1- beta2)*((delta_b[i])**2)
            m_hat_W = m_delta_W[i]/(1-beta1**t)
            v_hat_W = v_delta_W[i]/(1-beta2**t)
            m_hat_b = m_delta_b[i]/(1-beta1**t)
            v_hat_b = v_delta_b[i]/(1-beta2**t)
            W[i] = W[i] - (lr/(v_hat_W**(0.5)+epc))*(m_hat_W) 
            b[i] = b[i] - (lr/(v_hat_b**(0.5)+epc))*(m_hat_b)
    if step_counter%100 == 0:
        total_loss_train, total_error_train = calculate_total_loss_train()
        log_str_arr_train[log_str_arr_indx] = 'Epoch '+str(epoch)+', Step '+str(step_counter)+', Loss: '+str(total_loss_train)+', Error: '+str(round(total_error_train*100,2))+', lr: '+str(lr)
        total_loss_val, total_error_val = calculate_total_loss_val()
        log_str_arr_val[log_str_arr_indx] = 'Epoch '+str(epoch)+', Step '+str(step_counter)+', Loss: '+str(total_loss_val)+', Error: '+str(round(total_error_val*100,2))+', lr: '+str(lr)
        log_str_arr_indx+=1
    initialize_delta()
    return 0

def restore_weights(state):
    global W, b, tmp_delta_W, tmp_delta_b, m_delta_W, m_delta_b, v_delta_W, v_delta_b
    with open(save_dir +'weights_{}.pkl'.format(state), 'rb') as f:
              list_of_weights = pickle.load(f)
    W = list_of_weights[0]
    b = list_of_weights[1]
    if opt=='momentum' or opt=='nag':
        tmp_delta_W = list_of_weights[2]
        tmp_delta_b = list_of_weights[3]
    if opt=='adam':
        m_delta_W = list_of_weights[2]
        m_delta_b = list_of_weights[3]
        v_delta_W = list_of_weights[4]
        v_delta_b = list_of_weights[5]
    initialize_delta()
    return 0

def saveModelParams(epoch):
    list_of_weights = []
    list_of_weights.append(W)
    list_of_weights.append(b)
    if opt=='momentum' or opt=='nag':
        list_of_weights.append(tmp_delta_W)
        list_of_weights.append(tmp_delta_b)
    if opt=='adam':
        list_of_weights.append(m_delta_W)
        list_of_weights.append(m_delta_b)
        list_of_weights.append(v_delta_W)
        list_of_weights.append(v_delta_b)
        with open(save_dir +'weights_{}.pkl'.format(epoch), 'wb') as f:
            pickle.dump(list_of_weights, f)


def train_net():
    act_error = np.finfo(np.float64).max
    rep_epoch = 0
    global lr, step_counter, log_str_arr_indx, join_till, state, init_state
    epoch = int(init_state)
    while epoch<epochs:
        log_str_arr_indx = int((epoch*len(train_set))/(batch_size*100))
        n_batches = int(len(train_set)/batch_size)
        for batch in range(n_batches):
            for row in range(batch_size):
                indx = batch*batch_size + row
                forward_propagate(train_set[indx][0:num_input])
                actual_output = np.zeros(10)
                actual_output[int(train_set_op[indx])] = 1
                #loss = calculate_loss(actual_output)
                if opt=='nag':
                    for i in range(num_hidden+1):
                        W[i] = W[i] - momentum*(tmp_delta_W[i])
                        b[i] = b[i] - momentum*(tmp_delta_b[i])
                backward_propagate(actual_output, train_set[indx][0:num_input])
            update_weights(epoch, batch)
        if anneal=='true':
            loss,temp_error = calculate_total_loss_val()
            if temp_error>=act_error:
                if rep_epoch >max_rep_limit:
                    restore_weights(epoch)
                    join_till = int(epoch*(len(train_set)/(100*batch_size)))
                    break
                else:
                    rep_epoch +=1
                    restore_weights(epoch)
                    lr = lr/2
            else:
                rep_epoch = 0
                epoch+=1
                saveModelParams(epoch)
                act_error = temp_error
        else:
            epoch+=1
            loss,temp_error = calculate_total_loss_val()
            saveModelParams(epoch)

def predict_ex(test_data):
    len_test_data = len(test_data)
    predict_label = [int]*len_test_data
    for i in range(len_test_data):
        forward_propagate(test_data[i][:])
        predict_label[i] = np.argmax(outputsM[num_hidden])
    return predict_label
    return 0

def write_log_files():
    str_1 = '\n'.join(log_str_arr_train[0:join_till])
    str_2 = '\n'.join(log_str_arr_val[0:join_till])
    text_file = open(expt_dir+'log_train.txt', "w")
    text_file.write(str_1)
    text_file.close()
    text_file = open(expt_dir+'log_val.txt', "w")
    text_file.write(str_2)
    text_file.close()


parser = argparse.ArgumentParser()
parser.add_argument("--lr")
parser.add_argument("--momentum")
parser.add_argument("--num_hidden")
parser.add_argument("--sizes", type=str)
parser.add_argument("--activation")
parser.add_argument("--loss")
parser.add_argument("--opt")
parser.add_argument("--batch_size")
parser.add_argument("--epochs")
parser.add_argument("--anneal")
parser.add_argument("--save_dir")
parser.add_argument("--expt_dir")
parser.add_argument("--train")
parser.add_argument("--val")
parser.add_argument("--test")
parser.add_argument("--pretrain")
parser.add_argument("--state")
parser.add_argument("--testing")
args = parser.parse_args()
num_input = 784
num_output = 10
normalize_factor = np.float64(255.0)
max_rep_limit = 6
step_counter = 0
log_str_arr_indx = 0
if args.lr!=None:
    lr = float(args.lr)
if args.momentum!=None:
    momentum = float(args.momentum)
if args.num_hidden!=None:
    num_hidden = int(args.num_hidden)
if args.sizes:
    sizes = [int(xx) for xx in args.sizes.split(",")]
if args.activation!=None:
    activation = (args.activation).lower()
if args.loss!=None:
    loss_type = (args.loss).lower()
if args.opt!=None:
    opt = (args.opt).lower()
if args.batch_size!=None:
    batch_size = int(args.batch_size)
if args.epochs!=None:
    epochs = int(args.epochs)
if args.anneal!=None:
    anneal = (str(args.anneal)).lower()
save_dir = args.save_dir
expt_dir = args.expt_dir
test = args.test
pretrain = args.pretrain
state = args.state
testing = args.testing
if pretrain!=None:
    pretrain = (pretrain).lower()
if state!=None:
    state = int(state)
if testing!=None:
    testing = (testing).lower()    


# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# if not os.path.exists(expt_dir):
#     os.makedirs(expt_dir)

#48
W = [None]*(num_hidden+1)
b = [None]*(num_hidden+1)
delta_W = [None]*(num_hidden+1)
delta_b = [None]*(num_hidden+1)
outputsM = [None]*(num_hidden+1)
activationM = [None]*(num_hidden+1)
tmp_delta_W = [None]*(num_hidden+1)
tmp_delta_b = [None]*(num_hidden+1)
m_delta_W = [None]*(num_hidden+1)
m_delta_b = [None]*(num_hidden+1)
v_delta_W = [None]*(num_hidden+1)
v_delta_b = [None]*(num_hidden+1)

if testing=='true':
    restore_weights(state)
    test_set = np.array([list(map(np.float64,row) ) for row in  load_csv(test)])
    test_size = test_set.shape[0]
    predicted_label = predict_ex(test_set)
    id_vector = np.arange(len(predicted_label))
    add_to_csv = np.column_stack((id_vector, predicted_label))
    f = open(expt_dir+"predictions_{}.csv".format(state),"w")
    f.write('id,label\n')
    np.savetxt(f, add_to_csv.astype(int), fmt='%i', delimiter=",")
    f.close()
else:
    train = args.train
    val = args.val
    train_set = np.array([list(map(np.float64,row[1:]) ) for row in  load_csv(args.train)[1:]])
    validation_set = np.array([list(map(np.float64,row[1:]) ) for row in  load_csv(args.val)[1:]])
    test_set = np.array([list(map(np.float64,row[1:]) ) for row in  load_csv(args.test)[1:]])


    train_size = train_set.shape[0]
    val_size = validation_set.shape[0]
    test_size = test_set.shape[0]

    train_set_op = train_set[:,784]
    validation_set_op = validation_set[:,784]
    train_set = train_set[:,0:784]
    validation_set = validation_set[:,0:784]
    normalize_data()
    pca_set = np.concatenate((train_set,validation_set,test_set))
    temp_train = PCA(n_components=48, random_state=21)
    temp_train.fit(pca_set)
    num_input = temp_train.n_components_
    pca_set = temp_train.transform(pca_set)
    train_set = pca_set[0:train_size,:]
    validation_set = pca_set[train_size:train_size+val_size,:]
    test_set = pca_set[train_size+val_size:train_size+val_size+test_size,:]
    initialize_delta()
    if pretrain=='true':
        restore_weights(state)
        init_state=state
    else:
        initialize_net()
        init_state=0
    log_str_arr_train = ["" for x in range(int((len(train_set)*epochs)/(100*batch_size)))]
    log_str_arr_val = ["" for x in range(int((len(train_set)*epochs)/(100*batch_size)))]
    join_till = int(epochs*(len(train_set)/(100*batch_size)))
    train_net()
    write_log_files()
    predicted_label = predict_ex(test_set)
    id_vector = np.arange(len(predicted_label))
    add_to_csv = np.column_stack((id_vector, predicted_label))
    if state!=None:
        f = open(expt_dir+"predictions_{}.csv".format(state),"w")
    else:
        f = open(expt_dir+"predictions.csv","w")
    f.write('id,label\n')
    np.savetxt(f, add_to_csv.astype(int), fmt='%i', delimiter=",")
    f.close()
