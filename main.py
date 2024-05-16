from flask import Flask, render_template, request, redirect
import speech_recognition as sr 
from flask import jsonify
import pickle
app = Flask(__name__, template_folder='templates', static_folder='static')
extracted_terms = []
model1=pickle.load(open('model.pkl','rb'))

l1=['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction','Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
'Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis','Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

@app.route('/', methods=['GET', 'POST'])
def appp():
    transcript = ""

    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            recognizer = sr.Recognizer()
            audio_file = sr.AudioFile(file)
            
            with audio_file as source:
                data = recognizer.record(source)
            
            transcript = recognizer.recognize_google(data, key=None)
            print("Audio Transcription done")


            # Check for medical terms in the transcript
   
    return render_template('index.html', transcript=transcript)

@app.route('/process_terms', methods=['POST','GET'])
def process_terms():
    global extracted_terms
    data = request.get_json()
    terms = data.get("terms", [])
    
    # Check if terms are extracted
    if terms:
        print("Extracted Terms:", terms)  # Print extracted terms in the console
        extracted_terms.extend(terms)

        l2 = [1 if term in extracted_terms else 0 for term in l1] 
        input_data = [l2]
        predict = model1.predict(input_data)
        predicted = predict[0]
        
        predicted_probabilities = model1.predict_proba(input_data)[0]


        for a in range(0, len(disease)):
            if predicted == a:
                print(f"The predicted disease is: {disease[a]}")

        # Print the entire predicted_probabilities array and its shape
        print("Predicted Probabilities:", predicted_probabilities)
        print("Shape of predicted_probabilities:", predicted_probabilities.shape)
        
        results = [{"disease": disease[i], "probability": predicted_probabilities[i]*100} for i in range(len(disease))]
        
        results = sorted(results, key=lambda x: x["probability"], reverse=True)
        results = results[:5]
        for result in results:
            result['probability'] = round(result['probability'], 2)

            print(f"Predicted Disease: {result['disease']}, Probability: {result['probability']:.2f}%")

        return jsonify({"success": True, "results" : results})
    else:
        return jsonify({"success": False, "error": "No terms extracted"})


if __name__ == '__main__':
    app.run(debug=True)