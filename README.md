# IA App for CNN classification with fastapi

## Description
This project is a web application that allows users to upload images and classify them using a pre-trained CNN model. The application is built using FastAPI.


## Project Structure
```.
├──app
│   ├── data
│   │   ├── prices
│   │   │   ├── A6.csv
│   │   │   ├── classe_E.csv
│   │   │   ├── RAV4.csv
│   │   │   ├── Serie_1_2015_2017.csv
│   │   │   └── Serie_3.csv
│   │   └── specifications
│   │       └── data_full.csv
│   ├── model
│   │   └──  cars_finetunig.keras
│   ├── notebooks
│   │   ├── DL_car_recognition_finetuning.ipynb
│   │   └── ML_car_price_prediction.ipynb
│   ├── rapport
│   │   └── Livret d'apprentissage RNCP38616 - François ROY - Alyra - Promotion Berners Lee (2025).pdf
│   ├── __init__.py
│   ├── includes.py
│   ├── main.py
│   ├── model.py
│   └── preprocess.py
├── Dockerfile
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation
1. Clone the repository:
    ```bash
    git clone git@github.com:fry0035/projet-alyra
    cd projet_alyra
    ```

2. Install the docker image :
    ```bash
    docker build -t cars_app .
    ```

3. Run the docker container:
    ```bash
    docker run -p 8000:8000 cars_app
    ```

4. Open your browser and go to `http://localhost:8000` to access the FastAPI.


## Usage
**FastAPI Interface**:
   - Navigate to `http://localhost:8000/docs` to access the Swagger UI.
   - You can fill in the fields and upload an image to get the results, with the two text fields :
     - energy : 'Essence', 'Diesel', 'Hybride', 'Electrique'
     - transmission : 'Manuelle', 'Automatique'
