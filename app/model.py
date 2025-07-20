# For DL
import tensorflow as tf

# For ML
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .includes import MAP_CAR_PRICES_FILES

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def predict_image(model, image):
    prediction = model.predict(image)
    return prediction.tolist() # car on a 10 classes et on veut la liste des predictions de chaque classe


## ML part
def train_test_split_predict(model: object, X, y):
    # split data into training and validation data, for both features and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    # Define model
    car_model = model()

    # Fit model
    car_model.fit(train_X, train_y)

    # get predicted prices on validation data
    val_predictions = car_model.predict(val_X)
    # print(mean_absolute_error(val_y, val_predictions))

    return {'model': car_model, 'val_y': val_y, 'val_pred': val_predictions}

def ml_processing(car_data, year, km, energy, gb, power):
    # Select features
    X = car_data.drop(['Prix_occasion'], axis=1)
    y = car_data.Prix_occasion
    # Select categorical columns
    categorical_cols = [cname for cname in X.columns if
                        X[cname].dtype == "object"]
    # Select numerical columns
    numerical_cols = [cname for cname in X.columns if
                      X[cname].dtype in ['int64', 'float64']]
    # Keep only the numerical and categorical columns
    my_cols = categorical_cols + numerical_cols
    X = X[my_cols]
    # Check the number of missing values in each column
    missing_values = X.isnull().sum()
    # Keep only the columns with missing values
    missing_values = missing_values[missing_values > 0]

    # Preprocessing to add 2 more columns : Carburant, Boite_vitesse
    map_energy = {'Essence': 0, 'Gaz': 0, 'Diesel': 1, 'Hybride': 2, 'Electrique': 3}
    car_data.Carburant = [map_energy[en] for en in car_data.Carburant]

    map_gb = {'Manuelle': 0, 'Automatique': 1}
    car_data.Boite_vitesse = [map_gb[gb] for gb in car_data.Boite_vitesse]

    # Filter rows with missing price values
    filtered_car_data = car_data.dropna(axis=0)

    # Choose target and features
    y = filtered_car_data.Prix_occasion

    car_features = ['Annee_modele', 'Km', 'Carburant', 'Boite_vitesse', 'Puissance']
    X = filtered_car_data[car_features]

    # Linear regression
    result = train_test_split_predict(LinearRegression, X, y)
    car_model = result['model']

    return car_model.predict([[int(year), int(km), int(map_energy[energy]), int(map_gb[gb]), int(power)]])

def predict_price(car_model, year, km, energy, gb, power):


    try:
        car_file_path = MAP_CAR_PRICES_FILES[car_model]
    except Exception as e:
        print(f"str({e}): Données non disponibles pour ce modèle")
        return 0

    car_data = pd.read_csv(car_file_path)
    return ml_processing(car_data, year, km, energy, gb, power)

