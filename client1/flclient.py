"""FL client"""
import flwr as fl
import pickle
import client_utils as utils
import sys
import warnings
import datetime as dt
import socket
from glob import glob
import pandas as pd

if not sys.warnoptions:
    warnings.simplefilter("ignore")

IP = socket.gethostbyname(socket.gethostname()) 
PERSONALIZATION_COEF = 1 
GLOBAL_MODEL = r".\global_model.pkl"
TODAY = dt.datetime.today()
DATETIME = TODAY.strftime("%d-%m-%Y %H:%M:%S")
NB_ROUNDS = 5


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, len_f, len_nf, id, coef) -> None:
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.len_f = len_f
        self.len_nf = len_nf
        self.id = id
        self.coef = coef
        self.rnd = 1
        self.local_model_params = []


    def get_parameters(self, config):
        return utils.get_model_params(self.model)

    def fit(self, parameters, config):
        
        self.model = utils.set_model_params(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)
        # keep local model for personalization
        self.local_model_params = utils.get_model_params(self.model)
        loss = utils.get_loss(self.model, self.X_test, self.y_test)
        self.rnd += 1
        return utils.get_model_params(self.model), len(self.X_train), {"Local loss": loss}

    def evaluate(self, parameters, config):
        # Update local model with personalized parameters
        if parameters == []:
            personalized_params = self.local_model_params
        else:
            personalized_params = self.coef*self.local_model_params[0]  + (1 - self.coef)*parameters[0]

        personalized_model = utils.set_model_params(self.model, personalized_params)
    
        personalized_loss = utils.get_loss(personalized_model, self.X_test, self.y_test)
        f1 = utils.get_f1_score(personalized_model, self.X_test, self.y_test)

        if self.rnd == NB_ROUNDS:
            new_row = {
                'Datetime': self.id, 
                'Fall': self.len_f, 
                'Nonfall': self.len_nf, 
                'F1score': f1,
                'Loss': personalized_loss,
                'Personalisation(%)': 100*self.coef,
            }
            csv = glob(".\*.csv")
            if len(csv) > 0:
                df = pd.read_csv(csv[0])
                df = df._append(new_row, ignore_index=True)
                df.to_csv(csv[0], index=False)
            
            else:
                df = pd.DataFrame(new_row, index=[0])
                df.to_csv('.\history.csv', index=False)

        # Save global model
        pickle.dump(personalized_model, open(GLOBAL_MODEL, 'wb'))
        return personalized_loss, len(self.X_test), {"F1score": f1}


def listen_and_participate(serverPrams):

    url = f"{serverPrams['ip']}:{serverPrams['port']}"
    # Initialize model
    model = utils.prepare_model()

    # Load dataset
    X_train, y_train, X_test, y_test, len_f, len_nf = utils.load_datasets()

    if (len(X_train) > 0) and (len(y_train) > 0) and (len(X_test) > 0) and (len(y_test) > 0):

        # Instanciate Flower client
        fl_cli = FlowerClient(model, X_train, y_train, X_test, y_test, len_f, len_nf, DATETIME, serverPrams['personalization_coef'])

        # Start flower client
        fl.client.start_numpy_client(
            server_address=url,
            client=fl_cli,
            grpc_max_message_length=serverPrams['grpc_length']
        )
        response = "The training session was completed successfully"
    else:
        response = "Collect more fall and nonfall data"

    return response

if __name__ == '__main__':

    params = {
        'grpc_length': 1024*1024*1024,
        'nb_rounds': 5,
        'ip': IP, 
        'port': 5001,
        'personalization_coef': PERSONALIZATION_COEF
    }
    response = listen_and_participate(params)
    print(response)
