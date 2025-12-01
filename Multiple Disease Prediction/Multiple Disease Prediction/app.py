from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd


app = Flask(__name__)

with open('diabetes_model.pkl', 'rb') as file:
    diabetes_model = pickle.load(file)

with open('heart_disease_model.pkl', 'rb') as file:
    heart_model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes_predict', methods=['POST'])
def diabetes_predict():
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['bloodPressure'])
    skin_thickness = float(request.form['skinThickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetesPedigreeFunction'])
    age = float(request.form['age'])

    diabetes_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
    diabetes_prediction = diabetes_model.predict([diabetes_features])[0]

    if diabetes_prediction == 1:
        output_message = "You are predicted to have diabetes."
    else:
        output_message = "You are predicted to be healthy without diabetes."

    return render_template('result.html', prediction=output_message)


@app.route('/heart_predict', methods=['POST'])
def heart_predict():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    thal = float(request.form['thal'])

    heart_features = [age, sex, cp, chol, fbs, thalach, exang, oldpeak, slope, thal]
    heart_prediction = heart_model.predict([heart_features])[0]

    if heart_prediction == 1:
        output_message = "You are predicted to have heart disease."
    else:
        output_message = "You are predicted to be healthy without heart disease."

    return render_template('result.html', prediction=output_message)


class Disease_Prediction:
    Dis_Encoder={'abdominal_pain': 1,'abnormal_menstruation': 2,'acidity': 3,'acute_liver_failure': 4,'altered_sensorium': 5,'anxiety': 6,'back_pain': 7,'belly_pain': 8,'blackheads': 9,'bladder_discomfort': 10,'blister': 11,'blood_in_sputum': 12,'bloody_stool': 13,'blurred_and_distorted_vision': 14,'breathlessness': 15,'brittle_nails': 16,'bruising': 17,'burning_micturition': 18,'chest_pain': 19,'chills': 20,'cold_hands_and_feets': 21,'coma': 22,'congestion': 23,'constipation': 24,'continuous_feel_of_urine': 25,'continuous_sneezing': 26,'cough': 27,'cramps': 28,'dark_urine': 29,'dehydration': 30,'depression': 31,'diarrhoea': 32,'dischromic patches': 33,'distention_of_abdomen': 34,'dizziness': 35,'drying_and_tingling_lips': 36,'enlarged_thyroid': 37,'excessive_hunger': 38,'extra_marital_contacts': 39,'family_history': 40,'fast_heart_rate': 41,'fatigue': 42,'fluid_overload': 43,'fluid_overload.1': 44,'foul_smell_of urine': 45,'headache': 46,'high_fever': 47,'hip_joint_pain': 48,'history_of_alcohol_consumption': 49,'increased_appetite': 50,'indigestion': 51,'inflammatory_nails': 52,'internal_itching': 53,'irregular_sugar_level': 54,'irritability': 55,'irritation_in_anus': 56,'itching': 57,'joint_pain': 58,'knee_pain': 59,'lack_of_concentration': 60,'lethargy': 61,'loss_of_appetite': 62,'loss_of_balance': 63,'loss_of_smell': 64,'malaise': 65,'mild_fever': 66,'mood_swings': 67,'movement_stiffness': 68,'mucoid_sputum': 69,'muscle_pain': 70,'muscle_wasting': 71,'muscle_weakness': 72,'nausea': 73,'neck_pain': 74,'nodal_skin_eruptions': 75,'obesity': 76,'pain_behind_the_eyes': 77,'pain_during_bowel_movements': 78,'pain_in_anal_region': 79,'painful_walking': 80,'palpitations': 81,'passage_of_gases': 82,'patches_in_throat': 83,'phlegm': 84,'polyuria': 85,'prognosis': 86,'prominent_veins_on_calf': 87,'puffy_face_and_eyes': 88,'pus_filled_pimples': 89,'receiving_blood_transfusion': 90,'receiving_unsterile_injections': 91,'red_sore_around_nose': 92,'red_spots_over_body': 93,'redness_of_eyes': 94,'restlessness': 95,'runny_nose': 96,'rusty_sputum': 97,'scurring': 98,'shivering': 99,'silver_like_dusting': 100,'sinus_pressure': 101,'skin_peeling': 102,'skin_rash': 103,'slurred_speech': 104,'small_dents_in_nails': 105,'spinning_movements': 106,'spotting urination': 107,'stiff_neck': 108,'stomach_bleeding': 109,'stomach_pain': 110,'sunken_eyes': 111,'sweating': 112,'swelled_lymph_nodes': 113,'swelling_joints': 114,'swelling_of_stomach': 115,'swollen_blood_vessels': 116,'swollen_extremeties': 117,'swollen_legs': 118,'throat_irritation': 119,'toxic_look_(typhos)': 120,'ulcers_on_tongue': 121,'unsteadiness': 122,'visual_disturbances': 123,'vomiting': 124,'watering_from_eyes': 125,'weakness_in_limbs': 126,'weakness_of_one_body_side': 127,'weight_gain': 128,'weight_loss': 129,'yellow_crust_ooze': 130,'yellow_urine': 131,'yellowing_of_eyes': 132,'yellowish_skin': 133}
    Out_Dict={'(vertigo) Paroymsal  Positional Vertigo': 0,'AIDS': 1,'Acne': 2,'Alcoholic hepatitis': 3,'Allergy': 4,'Arthritis': 5,'Bronchial Asthma': 6,'Cervical spondylosis': 7,'Chicken pox': 8,'Chronic cholestasis': 9,'Common Cold': 10,'Dengue': 11,'Diabetes ': 12,'Dimorphic hemmorhoids(piles)': 13,'Drug Reaction': 14,'Fungal infection': 15,'GERD': 16,'Gastroenteritis': 17,'Heart attack': 18,'Hepatitis B': 19,'Hepatitis C': 20,'Hepatitis D': 21,'Hepatitis E': 22,'Hypertension ': 23,'Hyperthyroidism': 24,'Hypoglycemia': 25,'Hypothyroidism': 26,'Impetigo': 27,'Jaundice': 28,'Malaria': 29,'Migraine': 30,'Osteoarthristis': 31,'Paralysis (brain hemorrhage)': 32,'Peptic ulcer diseae': 33,'Pneumonia': 34,'Psoriasis': 35,'Tuberculosis': 36,'Typhoid': 37,'Urinary tract infection': 38,'Varicose veins': 39,'hepatitis A': 40}
    Pred_list = [[key, value] for key, value in Out_Dict.items()]
    Pred_Dict = {}
    for Pr_lis in Pred_list:
        Pred_Dict[Pr_lis[1]]=Pr_lis[0]
    def __init__(Self,n,s):
        Self.S=s
        Self.N=n
        Str=f'Health_Disease_M{Self.N}.csv'
        Self.Sym_dat=pd.read_csv(Str)
        Self.X=Self.Sym_dat.drop(['Unnamed: 0','prognosis'],axis=1)
        Self.Y=Self.Sym_dat['prognosis']
        Self.S=pd.DataFrame(Self.S)
        try:
            Self.S=Self.S.replace(Self.Dis_Encoder)
        except KeyError:
            print("Value is Incorrectly Entered")
            main()
    def classify(Self):
        model=RandomForestClassifier().fit(Self.X,Self.Y)
        pred=model.predict(Self.S)
        return Self.Pred_Dict[pred[0]]


@app.route('/predict_disease', methods=['POST'])
def predict_disease_route():
    symptom_count = int(request.form['symptomCount'])
    symptoms = [request.form[f'symptom{i+1}'] for i in range(symptom_count)]

    syms=[]
    syms.append(symptoms)
    User_pred=Disease_Prediction(symptom_count,syms)
    pred_out=User_pred.classify()

    return render_template('result.html', prediction=pred_out)


if __name__ == '__main__':
    app.run(debug=True)
