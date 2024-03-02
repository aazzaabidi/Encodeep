import argparse
import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.utils import shuffle
from models import InceptionTime, MultiRocket, ResNet50, RandomForest, TempCNN
from pandas import DataFrame
import seaborn as sns

def getBatch(array, i, batch_size):
    start_id = i * batch_size
    end_id = min((i + 1) * batch_size, array.shape[0])
    return array[start_id:end_id]

def train(model, x_train, y_train, loss_function, optimizer, batch_size, n_epochs, x_valid=None, y_valid=None, checkpoint_path=None):
    for e in range(n_epochs):
        x_train, y_train = shuffle(x_train, y_train)
        start = time.time()
        tot_loss = 0

        for ibatch in range(x_train.shape[0] // batch_size + 1):
            batch_x = getBatch(x_train, ibatch, batch_size)
            batch_y = getBatch(y_train, ibatch, batch_size)
            with tf.GradientTape() as tape:
                predictions = model(batch_x)
                loss = loss_function(batch_y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tot_loss += loss

        end = time.time()
        elapsed = end - start

        if x_valid is not None and y_valid is not None:
            valid_pred = model.predict(x_valid, batch_size=batch_size)
            valid_pred = np.argmax(valid_pred, axis=1)
            fscore = f1_score(y_valid, valid_pred, average='weighted')

            if fscore > best_valid_fMeasure:
                best_valid_fMeasure = fscore
                model.save_weights(checkpoint_path)
                print('saved')

            print("Epoch %d with loss %f and F-Measure on validation %f in %f seconds" % (e, tot_loss, fscore, elapsed))
            print(f1_score(y_valid, valid_pred, average=None))
        else:
            print("Epoch %d with loss %f in %f seconds" % (e, tot_loss, elapsed))

def main(args):
    if args.model == 'InceptionTime':
        model = InceptionTime()
    elif args.model == 'MultiRocket':
        model = MultiRocket()
    elif args.model == 'ResNet50':
        model = ResNet50()
    elif args.model == 'RandomForest':
        model = RandomForest()
    elif args.model == 'TempCNN':
        model = TempCNN()
    else:
        print("Invalid model name! Please choose from: 'InceptionTime', 'MultiRocket', 'ResNet50', 'RandomForest', 'TempCNN'")
        return

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    checkpoint_path = "checkpoint"

    # Assuming x_train, y_train, x_valid, y_valid are defined elsewhere
    train(model, x_train, y_train, loss_function, optimizer, args.batch_size, args.epochs, x_valid, y_valid, checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, help='Name of the model to use', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=34)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    args = parser.parse_args()
    main(args)
