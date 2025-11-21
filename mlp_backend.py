import numpy as np
import pandas as pd
from network_init import initialize_network 
from preprocessing import split_the_species, fit_preprocessor, transform_preprocessor
from MLP import MLP

class MLPBackend:
    def __init__(self):
        self.network = None
        self.mlp = None
        self.is_trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.fitted_preprocessor = None
        self.training_epochs_completed = 0
        self.num_classes = 3
        self.network_config = None
    
    def initialize_network(self, hidden_layers, hidden_neurons, learning_rate, activation_function, use_bias):
        self.network_config = {
            'hidden_layers': hidden_layers,
            'hidden_neurons': hidden_neurons,
            'learning_rate': learning_rate,
            'activation_function': activation_function,
            'use_bias': use_bias
        }
        self.network = initialize_network(**self.network_config)
        self.mlp = None
        self.is_trained = False
        self.training_epochs_completed = 0
    
    def Preprocessing(self):
        self.train_df, self.test_df = split_the_species()
        self.fitted_preprocessor, self.train_df = fit_preprocessor(self.train_df)
        self.test_df = transform_preprocessor(self.test_df, self.fitted_preprocessor)

        self.X_train = self.train_df.drop('Species', axis=1)
        self.X_test = self.test_df.drop('Species', axis=1)
        self.y_train = self.train_df['Species']
        self.y_test = self.test_df['Species']
        
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)
    
    def train_network(self, epochs=100):
        try:
            self.Preprocessing()
            
            if self.network_config is None:
                raise Exception("Network configuration not set. Initialize the network first.")

            self.network = initialize_network(**self.network_config)
            self.mlp = MLP(self.network)
            self.is_trained = False
            self.training_epochs_completed = 0
            
            self.mlp.train(self.X_train.values, self.y_train.values, epochs=epochs)
            self.training_epochs_completed += epochs
            self.is_trained = True
            
            y_train_pred = self.mlp.predict(self.X_train.values)
            train_accuracy = np.mean(y_train_pred == self.y_train.values)
            
            return train_accuracy
            
        except Exception as e:
            raise e
    
    def test_network(self):
        if not hasattr(self, 'mlp') or self.mlp is None or not self.is_trained:
            raise Exception("Please train the network first!")
        
        y_pred = self.mlp.predict(self.X_test.values)
        y_true = self.y_test.values
        confusion_mat = np.zeros((3, 3), dtype=int)
        
        for true, pred in zip(y_true, y_pred):
            confusion_mat[true, pred] += 1
        
        accuracy = np.trace(confusion_mat) / np.sum(confusion_mat)
        return {
            'confusion_mat': confusion_mat,
            'accuracy': accuracy,
            'test_samples': len(self.X_test),
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def classify_sample(self, features):
        if not hasattr(self, 'mlp') or self.mlp is None or not self.is_trained:
            raise Exception("Please train the network first!")
        
        sample_array = np.array(features).reshape(1, -1)
        sample_df = pd.DataFrame(sample_array, columns=self.X_train.columns)
        sample_df[self.fitted_preprocessor['scaler'].feature_names_in_] = \
            self.fitted_preprocessor['scaler'].transform(sample_df[self.fitted_preprocessor['scaler'].feature_names_in_])
        
        sample_processed = sample_df.values
        prediction = self.mlp.predict(sample_processed)[0]        
        sample_flat = sample_processed[0]
        _, activations = self.mlp.forward(sample_flat)
        probabilities = activations[-1]
        
        return {
            'predicted_class': prediction + 1,
            'probabilities': probabilities,
            'confidence': np.max(probabilities)
        }
    
    def get_decision_boundary_data(self):
        if not hasattr(self, 'mlp') or self.mlp is None or not self.is_trained or self.X_train is None:
            return None
        
        X_vis = self.X_train.iloc[:, :2].values
        y_vis = self.y_train.values        
        x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
        y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        if self.X_train.shape[1] > 2:
            feature_means = self.X_train.mean(axis=0).values
            padding = np.tile(feature_means[2:], (len(mesh_points), 1))
            mesh_points = np.hstack([mesh_points, padding])
        
        Z = self.mlp.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        return xx, yy, Z, X_vis, y_vis
    
    def get_training_info(self):
        return {
            'is_trained': self.is_trained,
            'epochs_completed': self.training_epochs_completed,
            'has_network': self.network is not None,
            'has_mlp': self.mlp is not None
        }