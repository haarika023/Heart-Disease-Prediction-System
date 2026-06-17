import pandas as pd #to handle dataset
from sklearn.model_selection import train_test_split #80,20
from sklearn.preprocessing import StandardScaler #normalize data into same range 
from sklearn.neural_network import MLPClassifier #classifies as yes/no ,0/1
from sklearn.metrics import accuracy_score #to check model performance

# ================= LOAD DATA =================
print("\n================ HEART DISEASE SYSTEM =================\n")

data = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv") #load dataset from file

print("Dataset Loaded Successfully ✅")
print(f"Total Records: {data.shape[0]}")
print(f"Total Features: {data.shape[1]}\n")


# ================= PREPROCESS =================
X = data.drop("HeartDiseaseorAttack", axis=1) #all inputs except target 
y = data["HeartDiseaseorAttack"] #target column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #learning
X_test = scaler.transform(X_test)       #testing

print("Data Preprocessing Completed ✅")


# ================= TRAIN ANN =================
model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300) 
model.fit(X_train, y_train)

print("ANN Model Training Completed ✅")


# ================= MODEL PERFORMANCE =================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) #actual vs predicted

print("\n====================================================")
print("           MODEL PERFORMANCE")
print("====================================================")
print(f"MODEL ACCURACY          : {round(accuracy*100,2)} %")
print("====================================================")


# ================= FUZZY SYSTEM =================
def fuzzy_adjustment(bp, bmi, age, diabetes, prob):
    risk = prob

    if bp == 1:
        risk += 0.1
    if bmi > 30:
        risk += 0.1
    if age > 8:
        risk += 0.1
    if diabetes == 1:
        risk += 0.15

    return max(0, min(1, risk))


# ================= TEST SAMPLE =================
sample = X.iloc[0]

bp = sample["HighBP"]
bmi = sample["BMI"]
age = sample["Age"]
diabetes = sample["Diabetes"]

sample_scaled = scaler.transform([sample])
ann_prob = model.predict_proba(sample_scaled)[0][1]

final_risk = fuzzy_adjustment(bp, bmi, age, diabetes, ann_prob)


# ================= FINAL OUTPUT =================
print("\n----------------------------------------------------")
print("      HEART DISEASE PREDICTION SYSTEM LOG")
print("----------------------------------------------------")

print(f"INPUT: BP={bp}, BMI={bmi}, AGE={age}, DIABETES={diabetes}")
print(f"ANN PREDICTION PROBABILITY     : {round(ann_prob,2)}")
print(f"FUZZY ADJUSTED RISK            : {round(final_risk,2)}")

if final_risk > 0.5:
    status = "HIGH RISK ⚠️"
else:
    status = "LOW RISK ✅"

print(f"FINAL STATUS                   : {status}")
print("NEXT CHECKUP                   : Recommended")
print("====================================================\n")
