import pandas as pd
import numpy as np
from keras.models import load_model




class MakePrediction():
    def __init__(self,TestDataLocation = "data/test.csv"):
        self.model = load_model('neuralnetwork_model_A_rmsprop.h5')
        #self.model = load_model('my_model_b_adam.h5')
        self.test_dataframe = pd.read_csv(TestDataLocation)
        #print(self.test_dataframe.head())

    def run_inference_model(self):
        self.feature_data = np.array(self.test_dataframe[self.test_dataframe.columns[0:14]])
        predicted_labels = self.model.predict(self.feature_data)
        #print(type(predicted_labels))
        with open("results.csv","w") as self.outfile:
            self.outfile.write("Winner\n")
            for i in predicted_labels:
                if round(float(i)) == 1:
                    self.outfile.write("Barack Obama\n")
                elif round(float(i))==0:
                    self.outfile.write("Mitt Romney\n")




if __name__ == '__main__':
    predicted_values = MakePrediction()
    predicted_values.run_inference_model()
