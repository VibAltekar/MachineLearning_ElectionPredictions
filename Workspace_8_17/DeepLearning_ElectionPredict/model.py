import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle
from keras.models import load_model
from sklearn.ensemble import GradientBoostingClassifier


class ElectionModel():
    def __init__(self, TrainingDataLocation = "data/train.csv"):
        self.training_data = pd.read_csv(TrainingDataLocation)
        self.feature_data = self.training_data[self.training_data.columns[0:14]]
        self.label_data_names = self.training_data[self.training_data.columns[14]]
        self.presidents = ["Mitt Romney","Barack Obama"]
        self.label_data = np.array([self.presidents.index(x) for x in self.label_data_names])
        self.training_data['Winners_Binary'] = pd.Series(self.label_data, index=self.training_data.index)
        print(self.training_data.head(10))
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(self.feature_data, self.training_data['Winners_Binary'], test_size=0.1)
        with open("log_perf.txt","a") as logfile:
            logfile.write("-"*50+ "\n")
        self.gbc_model = self.GradBoost()
        self.gbc_model.save("modelGB.h5")
        return None

    def NaiveBayes(self):
        model_name = "Gaussian Naive Bayes Classifer"
        self.NB_clf = GaussianNB()
        self.NB_clf.fit(np.array(self.train_features),np.array(self.train_labels))
        NB_clf_predictions = self.NB_clf.predict(np.array(self.test_features))
        #NaiveBayes_ACC = accuracy_score(self.test_labels,NB_clf_predictions)
        NaiveBayes_ACC = cross_val_score(self.NB_clf,self.feature_data,self.training_data['Winners_Binary'],cv=StratifiedKFold(n_splits=5,shuffle=True))
        print(NB_clf_predictions[0:5])
        print(self.test_labels[0:5])
        NB_ACC_ave = np.mean(NaiveBayes_ACC)
        print(str(model_name) + " " + str(NB_ACC_ave))
        self.Log_Model_Stats(model_name,metrics=NaiveBayes_ACC,model_accuracy=NB_ACC_ave)
        return self.NB_clf

    def SupportVectorMachine(self):
        model_name = "Support Vector Machine"
        self.SVM_model = svm.SVC()
        #self.SVM_model.fit(self.train_features,self.train_labels)
        #SVM_predictions = self.SVM_model.predict(self.test_features)
        #SVM_ACC = accuracy_score(self.test_labels,SVM_predictions)
        SVM_ACC = cross_val_score(self.SVM_model,self.feature_data,self.training_data['Winners_Binary'],cv=StratifiedKFold(n_splits=5))
        SVM_ACC_ave = np.mean(SVM_ACC)
        print(str(model_name) + " " + str(SVM_ACC_ave))
        self.Log_Model_Stats(model_name=model_name, model_accuracy=SVM_ACC_ave,metrics=SVM_ACC)
        return self.SVM_model


    def NeuralNetwork_Model_A(self, activation="relu",optimizer="rmsprop",train=True):
        model_name = "NeuralNetwork Model A"
        self.model_a = Sequential()
        self.model_a.add(Dense(64,input_dim=14, activation=activation))
        self.model_a.add(Dropout(0.5))
        self.model_a.add(Dense(64,activation=activation))
        self.model_a.add(Dropout(0.5))
        self.model_a.add(Dense(1, activation="sigmoid"))
        self.model_a.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
        if train==True:
            print(self.NNpredict(model_title=self.model_a))
            print("predicted ",self.model_a.predict((self.test_features)))
            print("truth ",self.test_labels)
            #self.Log_Model_Stats(model_name, final_training_acc,metrics=metrics,optimizer=optimizer,activation=activation)

            if optimizer == "rmsprop":
                self.model_a.save('my_model_a_rmsprop.h5')  # creates a HDF5 file 'my_model.h5'
            elif optimizer=="adam":
                self.model_a.save('my_model_a_adam.h5')
        return self.model_a

    def NeuralNetwork_Model_B(self, activation="tanh",optimizer="adam",train=True):
        model_name = "NeuralNetwork Model B"
        self.model_b = Sequential()
        self.model_b.add(Dense(128,input_dim=14,activation=activation))
        self.model_b.add(Dense(64, activation="relu"))
        self.model_c.add(Dropout(0.10))
        self.model_b.add(Dense(32, activation=activation))
        self.model_b.add(Dense(16, activation=activation))
        self.model_b.add(Dropout(0.15))
        self.model_b.add(Dense(8, activation=activation))
        self.model_b.add(Dense(1, activation="sigmoid"))
        self.model_b.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
        if train==True:
            history = self.model_b.fit(np.array(self.feature_data),np.array(self.training_data['Winners_Binary']),epochs=100,verbose=0,batch_size=20)
            final_training_acc = history.history["acc"][-1]
            metrics = self.model_b.evaluate(np.array(self.test_features),np.array(self.test_labels), batch_size=20)
            print("Accuracy with " + str(activation) + " " + str(optimizer) + " " + str(final_training_acc))
            #score = self.model_a.predict(np.array(self.test_features))
            #print("ACCSCORE ",accuracy_score(y_true=np.array(self.test_labels), y_pred=score))
            self.Log_Model_Stats(model_name, final_training_acc,metrics=metrics,optimizer=optimizer,activation=activation)
            if optimizer == "rmsprop":
                self.model_b.save('my_model_b_rmsprop.h5')  # creates a HDF5 file 'my_model.h5'
            elif optimizer=="adam":
                self.model_b.save('my_model_b_adam.h5')

        return self.model_b

    def Log_Model_Stats(self, model_name, model_accuracy,metrics,optimizer=" ",activation=" "):
        model_accuracy = round(model_accuracy,3)
        if len(optimizer) > 2 and len(activation) > 2:
            logput = " with " + optimizer + " " + activation
        else:
            logput = " "
        with open("log_perf.txt","a") as logfile:
            logfile.write(str(model_name) + logput + " resulted in training accuracy of " + str(model_accuracy)+" metrics: " + str(metrics) + "\n")
        return None
    def NNpredict(self,model_title):
        history = model_title.fit(self.train_features,self.train_labels,epochs=100,batch_size=20,verbose=0)
        metrics = model_title.evaluate(self.test_features,self.test_labels,verbose=0)
        return (history.history["acc"][-1],  history.history["loss"][-1] , metrics[1],metrics[0]          )


    def NeuralNetwork_Model_C(self, activation="sigmoid",optimizer="rmsprop",train=True):
        model_name = "Neural Network Model C"
        self.model_c = Sequential()
        self.model_c.add(Dense(128,input_dim=14,activation="relu"))
        self.model_c.add(Dropout(0.5))
        self.model_c.add(Dense(64, activation="relu"))
        self.model_c.add(Dropout(0.5))
        self.model_c.add(Dense(16, activation="relu"))
        self.model_c.add(Dropout(0.5))
        self.model_c.add(Dense(1, activation="sigmoid"))
        self.model_c.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
        if train==True:
            history = self.model_c.fit((self.train_features),np.array(self.train_labels),epochs=100,verbose=0,batch_size=20)
            metrics = self.model_c.evaluate(self.test_features,(self.test_labels),verbose=0, batch_size=20)
            print("----------------------------------------------------------------------------------------")
            print(history.history["acc"][-1])
            print("metrics ",metrics)
            print("predicted ",self.model_c.predict((self.test_features)))
            print(np.array(self.train_features).shape)
            print("truth ",np.array(self.test_labels))
            self.model_c.save("test_model.h5")
            print(np.array(self.train_labels))
            history = self.model_c.fit(np.array(self.train_features),np.array(self.train_labels),epochs=100,verbose=0,batch_size=20)
            final_training_acc = history.history["acc"][-1]
            metrics = self.model_c.evaluate(np.array(self.test_features),np.array(self.test_labels), batch_size=20)
            #print("Accuracy with " + str(activation) + " " + str(optimizer) + " " + str(final_training_acc))
            print("metrics ",metrics)
            score = self.model_c.predict(np.array(self.test_features))
            print("-------------------")
            print(score)
            print(self.test_labels)

            print("-------------------")
            print("ACCSCORE ",accuracy_score(y_true=np.array(self.test_labels), y_pred=np.array(score)))
            self.Log_Model_Stats(model_name, final_training_acc,metrics=metrics,optimizer=optimizer,activation=activation)
            if optimizer == "rmsprop":
                self.model_c.save('my_model_c_rmsprop.h5')  # creates a HDF5 file 'my_model.h5'
            elif optimizer=="adam":
                self.model_c.save('my_model_c_adam.h5')
        print(np.array(self.test_features)[0:5])
        print(np.array(self.test_labels)[0:5])
        print(self.model_c.predict(np.array(self.test_features)[0:5]))
        return self.model_c

    def GridSearch_NeuralNetworkTuning(self,model_name):
        print("Grid Search started")
        model = KerasClassifier(build_fn=model_name, epochs=100, batch_size=10, verbose=0)
        activation = ['softplus', 'relu', 'tanh', 'sigmoid']
        optimizer = ['adam','adadelta']
        param_grid = dict(activation=activation,optimizer=optimizer)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,cv=StratifiedKFold(n_splits=5))
        grid_result = grid.fit(np.array(Mx.feature_data), np.array(Mx.training_data['Winners_Binary']))
        print(" Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))



if __name__ == '__main__':
    #Mx = ElectionModel()
    #Mx.NaiveBayes()
    #Mx.SupportVectorMachine()
    #Mx.NeuralNetwork_Model_A(activation="relu",optimizer="adadelta",train=True)
    #Mx.NeuralNetwork_Model_A(activation="tanh",optimizer="rmsprop",train=True)

    #Mx.NeuralNetwork_Model_B(activation="tanh",optimizer="adam",train=True)
    #Mx.NeuralNetwork_Model_B(activation="tanh",optimizer="rmsprop",train=True)




    #Mx.NeuralNetwork_Model_B(activation="tanh")
    #Mx.NeuralNetwork_Model_B(activation="sigmoid")
    #Mx.GridSearch_NeuralNetworkTuning(model_name=Mx.NeuralNetwork_Model_A)
    #Mx.GridSearch_NeuralNetworkTuning(model_name=Mx.NeuralNetwork_Model_B)
    #Mx.GridSearch_NeuralNetworkTuning(model_name=Mx.NeuralNetwork_Model_B)
    #Mx.GridSearch_NeuralNetworkTuning(model_name=Mx.NeuralNetwork_Model_C)
