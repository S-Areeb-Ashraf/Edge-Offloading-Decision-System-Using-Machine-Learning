import tkinter as tk
import seaborn as sns
import numpy as np
import pandas as pd
from tkinter import messagebox
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#                       --- Cost functions + rule-based decision ---
def compute_cost_local(cpu,battery,latency,alpha=0.5,beta=0.5):
    energy=(100-battery)*0.2+cpu*0.8
    time=latency
    return alpha*energy+beta*time

def compute_cost_cloud(cpu,battery,latency,bandwidth,data_size,alpha=0.5,beta=0.5):
    energy = (100-battery)*0.4
    transfer_time=(data_size/max(bandwidth,1))*10  
    time = latency+transfer_time+5
    return alpha*energy+beta*time

def rule_based_decision(cpu,battery,latency,bandwidth,data_size,alpha=0.5,beta=0.5):
    cost_local=compute_cost_local(cpu,battery,latency,alpha,beta)
    cost_cloud=compute_cost_cloud(cpu,battery,latency,bandwidth,data_size,alpha,beta)
    return 0 if cost_local < cost_cloud else 1



# Synthtic data is generated and then use pandas to read the csv file

df=pd.read_csv("offloading_noisy_dataset_v2.csv")


#                                                   Pre-Processing the data
#                                   ***CPU usage*** 
cpu_val=df["CPU_Usage"].mean()    
df.loc[df["CPU_Usage"]==0,"CPU_Usage"]=cpu_val
df["CPU_Usage"]=df["CPU_Usage"].fillna(cpu_val)

#                                   ***Battery Level***
bat_val=df["Battery_Level"].mean()    
df.loc[df["Battery_Level"]==0,"Battery_Level"]=bat_val
df["Battery_Level"]=df["Battery_Level"].fillna(bat_val)

#                                   ***Network Latency***
net_val=df["Network_Latency"].mean()
df.loc[df["Network_Latency"]==0,"Network_Latency"]=net_val
df["Network_Latency"]=df["Network_Latency"].fillna(net_val)

#                                  ***Bandwidth Availability***
ban_val=df["Bandwidth_Availability"].mean()    
df.loc[df["Bandwidth_Availability"]==0,"Bandwidth_Availability"]=ban_val
df["Bandwidth_Availability"]=df["Bandwidth_Availability"].fillna(ban_val)

#                                      ***Data Size***
data_val=df["Data_Size"].mean()    
df.loc[df["Data_Size"]==0,"Data_Size"]=data_val
df["Data_Size"]=df["Data_Size"].fillna(data_val)


#                   Applying        ***** Encoding (Label Encoding) *****
#                       Using label encoding for the task complexity column (low,meduim,high)

df_l=df
le=LabelEncoder()
df_l["Task_Complexity"]=le.fit_transform(df_l["Task_Complexity"])


#                       Now using Standrd Sclaer to standardize the numerical features which influence the models
#                               Dropping target col when applying standard scaler
sd=StandardScaler()
df_main=df_l.drop("Decision",axis=1)
df_main=df_main.drop("User_ID",axis=1)
df_target=df["Decision"]


#                                   Selecting numerical cols to apply standard scalar
num_cols=df_main.select_dtypes(include=[np.number]).columns.tolist()
df_main[num_cols]=sd.fit_transform(df_main[num_cols])


smote=SMOTE(random_state=42)
X_resampled,y_resampled=smote.fit_resample(df_main,df_target)

#                       Spilitting the data set in 80% fpr training and 20% for testing
df_train,df_test,dfy_train,dfy_test=train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=42)


model_dt=DecisionTreeClassifier()
model_rf=RandomForestClassifier()
model_lr=LogisticRegression()
model_kn=KNeighborsClassifier()
model_nn=MLPClassifier(hidden_layer_sizes=(100,),max_iter=500,random_state=42)
model_sv=SVC()


#                                   Training all Models
model_dt.fit(df_train,dfy_train)
model_rf.fit(df_train,dfy_train)
model_lr.fit(df_train,dfy_train)
model_kn.fit(df_train,dfy_train)
model_nn.fit(df_train,dfy_train)
model_sv.fit(df_train,dfy_train)

# model_rf.

df_l=df_l.drop("Decision",axis=1)
df_l['Rule_Based_Decision']=df_l.apply(
    lambda row: rule_based_decision(
        row['CPU_Usage'], 
        row['Battery_Level'], 
        row['Network_Latency'],
        row['Bandwidth_Availability'],
        row['Data_Size'],
        alpha=0.3, beta=0.7
    ), axis=1
)


#                               Training Accuracy
acc_train=accuracy_score(dfy_train,model_dt.predict(df_train))


#                   Testing of  (Desicion Tree Classifier)

#                                               Accuracy Score
acc_score_d=accuracy_score(dfy_test,model_dt.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_dt.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_dt.classes_,columns=model_dt.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_dt.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File
# with open("Result_v1.md","w") as file:
#     file.write("\t\t\t\t"+"Desicion Tree Classifier"+"\n\n")
#     file.write("Accuracy Score: "+str(round(acc_score_d*100,2))+"\n\n")
#     file.write("Confusion Matrix"+"\n\n")
#     file.writelines(str(cm_df)+"\n\n")
#     file.write("Classification Report"+"\n\n")
#     file.writelines(str(report_df.round(2))+"\n\n")
with open("Result_v1.md", "w") as file:
    file.write("# Decision Tree Classifier Results\n\n")

    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_d * 100, 2)}%**\n\n")

    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")

    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")


#                               Training Accuracy
acc_train_r=accuracy_score(dfy_train,model_dt.predict(df_train))

#                   Testing of  (Random Forest)

#                                               Accuracy Score
acc_score_r=accuracy_score(dfy_test,model_rf.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_rf.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_rf.classes_,columns=model_rf.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_rf.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File
# with open("Result_v1.md","a") as file:
#     file.write("\t\t\t\t"+"Random Forests"+"\n\n")
#     file.write("Accuracy Score: "+str(round(acc_score_r*100,2))+"\n\n")
#     file.write("Confusion Matrix"+"\n\n")
#     file.writelines(str(cm_df)+"\n\n")
#     file.write("Classification Report"+"\n\n")
#     file.writelines(str(report_df.round(1))+"\n\n")

with open("Result_v1.md","a") as file:
    file.write("# Random Forest Results\n\n")

    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_r * 100, 2)}%**\n\n")

    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")

    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")




#                               Plotting Confusion Matrix for Random forest
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")


#                               PLotting Bargraph for Classification Report
report_df=pd.DataFrame(report).transpose().drop(['accuracy','macro avg','weighted avg'])

report_df[['precision','recall','f1-score']].plot(kind='bar',figsize=(8,5))
plt.title('Classification Report Metrics')
plt.ylabel('Score')
plt.ylim(0,1.05)
plt.xticks(rotation=0)
plt.grid(axis='y')


#                               PLotting bargraph of Training and Testing Accuracy 
acc_types=['Train Accuracy','Test Accuracy']
acc_values=[acc_train_r,acc_score_r]

plt.figure(figsize=(5,4))
plt.bar(acc_types,acc_values,color=['skyblue','salmon'])
plt.title('Accuracy of Random Forests')
plt.ylabel('Accuracy')
plt.ylim(0.0,1.0)
plt.grid(axis='y',linestyle='--', alpha=0.7)
# plt.show()


#                   Testing of  (Logistic Regression)

#                                               Accuracy Score
acc_score_l=accuracy_score(dfy_test,model_lr.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_lr.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_lr.classes_,columns=model_lr.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_lr.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File
# with open("Result_v1.md","a") as file:
#     file.write("\t\t\t\t"+"Logistic Regression"+"\n\n")
#     file.write("Accuracy Score: "+str(round(acc_score_l*100,2))+"\n\n")
#     file.write("Confusion Matrix"+"\n\n")
#     file.writelines(str(cm_df)+"\n\n")
#     file.write("Classification Report"+"\n\n")
#     file.writelines(str(report_df.round(2))+"\n\n")

with open("Result_v1.md","a") as file:
    file.write("# Logistic Regression Results\n\n")

    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_l * 100, 2)}%**\n\n")

    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")

    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")


#                   Testing of  (SVC---Support Vector Machine)

#                                               Accuracy Score
acc_score_s=accuracy_score(dfy_test,model_sv.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_sv.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_sv.classes_,columns=model_sv.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_sv.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File
# with open("Result_v1.md","a") as file:
#     file.write("\t\t\t\t"+"Support Vector Classifier"+"\n\n")
#     file.write("Accuracy Score: "+str(round(acc_score_s*100,2))+"\n\n")
#     file.write("Confusion Matrix"+"\n\n")
#     file.writelines(str(cm_df)+"\n\n")
#     file.write("Classification Report"+"\n\n")
#     file.writelines(str(report_df.round(2))+"\n\n")

with open("Result_v1.md","a") as file:
    file.write("# Support Vector Classifier Results\n\n")

    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_s * 100, 2)}%**\n\n")

    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")

    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")

#                   Testing of  (KNN)

#                                               Accuracy Score
acc_score_k=accuracy_score(dfy_test,model_kn.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_kn.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_kn.classes_,columns=model_kn.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_kn.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File
# with open("Result_v1.md","a") as file:
#     file.write("\t\t\t\t"+"K-Nearest Neighbor"+"\n\n")
#     file.write("Accuracy Score: "+str(round(acc_score_k*100,2))+"\n\n")
#     file.write("Confusion Matrix"+"\n\n")
#     file.writelines(str(cm_df)+"\n\n")
#     file.write("Classification Report"+"\n\n")
#     file.writelines(str(report_df.round(2))+"\n\n")

with open("Result_v1.md","a") as file:
    file.write("# K-Nearest Neighbor Results\n\n")

    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_k * 100, 2)}%**\n\n")

    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")

    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")



#                   Testing of  (MLP Classifier ---Neural Networks)

#                                               Accuracy Score
acc_score_n=accuracy_score(dfy_test,model_nn.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_nn.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_nn.classes_,columns=model_nn.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_nn.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File
# with open("Result_v1.md","a") as file:
#     file.write("\t\t\t\t"+"Neural Networks (MLP Classifier)"+"\n\n")
#     file.write("Accuracy Score: "+str(round(acc_score_n*100,2))+"\n\n")
#     file.write("Confusion Matrix"+"\n\n")
#     file.writelines(str(cm_df)+"\n\n")
#     file.write("Classification Report"+"\n\n")
#     file.writelines(str(report_df.round(2))+"\n\n")



with open("Result_v1.md","a") as file:
    file.write("# Neural Networks (MLP Classifier) Results\n\n")

    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_n * 100, 2)}%**\n\n")

    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")

    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")


y_pred_rule = df_test.apply(
    lambda row: rule_based_decision(
        row['CPU_Usage'],
        row['Battery_Level'],
        row['Network_Latency'],
        row['Bandwidth_Availability'],
        row['Data_Size']
    ), axis=1
)


#               Rule-Based Evaluation        (Scores on Mathematical Equation)
acc_rule = accuracy_score(dfy_test,y_pred_rule)
cm_rule = confusion_matrix(dfy_test,y_pred_rule)
report_rule = classification_report(dfy_test,y_pred_rule,output_dict=True)
report_df_rule = pd.DataFrame(report_rule).transpose()


#                                       Reporting Scorees to File
# with open("Result_v1.md","a") as file:
#     file.write("\t\t\t\t"+"Rule-Based Decision"+"\n\n")
#     file.write("Accuracy Score: "+str(round(acc_rule*100,2))+"\n\n")
#     file.write("Confusion Matrix\n\n")
#     file.writelines(str(pd.DataFrame(cm_rule)) + "\n\n")
#     file.write("Classification Report\n\n")
#     file.writelines(str(report_df_rule.round(2)) + "\n\n")

with open("Result_v1.md","a") as file:
    file.write("# Rule-Based Decision Results\n\n")

    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_rule*100,2)}%**\n\n")

    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")

    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")


models=["Rule Based Performance","Decisicon Tree","Random Forest","Logistic Regression","SVC","K-NN","Neural Networks(MLP)"]
test_acc=[acc_rule,acc_score_d,acc_score_r,acc_score_l,acc_score_s,acc_score_k,acc_score_n]


#                                       Accuracy comparasion on diff models

plt.figure(figsize=(8,5))
plt.plot(models,test_acc,marker='o',linestyle='-',color='teal',label='Test Accuracy')
plt.title('Model Accuracy Comparison on Test Set')
plt.ylabel('Accuracy')
plt.ylim(0.45,1.0)
plt.grid(True,linestyle='--',alpha=0.6)
plt.xticks(rotation=15)
plt.legend()
plt.tight_layout()

#                               GUI for Model based input
root = tk.Tk()
root.title("Offloading Decision Predictor")
root.geometry("400x500")


entries = {}
fields = [
    "CPU Usage (%)",
    "Battery Level (%)",
    "Network Latency (ms)",
    "Bandwidth Availability (kbps)",
    "Data Size (KB)",
    "User ID"
]

#                               GUI for Rule based input
root_rule = tk.Tk()
root_rule.title("Offloading Decision Predictor (Rule-Based)")
root_rule.geometry("400x500")

entries_rule = {}
for field in fields:
    label = tk.Label(root_rule, text=field)
    label.pack()
    entry = tk.Entry(root_rule)
    entry.pack()
    entries_rule[field] = entry

task_complexity_var_rule = tk.StringVar(root_rule)
task_complexity_var_rule.set("medium")
tk.Label(root_rule, text="Task Complexity").pack()
tk.OptionMenu(root_rule, task_complexity_var_rule, "low", "medium", "high").pack()


#                       Rule based Prediction

def predict_rule_decision():
    try:
        cpu = float(entries_rule["CPU Usage (%)"].get())
        battery = float(entries_rule["Battery Level (%)"].get())
        latency = float(entries_rule["Network Latency (ms)"].get())
        bandwidth = float(entries_rule["Bandwidth Availability (kbps)"].get())
        data_size = float(entries_rule["Data Size (KB)"].get())

        decision = rule_based_decision(cpu, battery, latency, bandwidth, data_size)
        result = "Offload to Cloud" if decision == 1 else "Execute Locally"
        messagebox.showinfo("Rule-Based Decision", f"Decision: {result}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

predict_btn_rule = tk.Button(root_rule, text="Predict Rule-Based Decision", command=predict_rule_decision)
predict_btn_rule.pack(pady=20)

# model
#                              Model based Prediction
def predict_decision():
    try:
        input_data = {
            'CPU_Usage': float(entries["CPU Usage (%)"].get()),
            'Battery_Level': float(entries["Battery Level (%)"].get()),
            'Network_Latency': float(entries["Network Latency (ms)"].get()),
            'Bandwidth_Availability': float(entries["Bandwidth Availability (kbps)"].get()),
            'Data_Size': float(entries["Data Size (KB)"].get()),
            'User_ID': int(entries["User ID"].get()),
            'Task_Complexity': task_complexity_var.get()
        }

        df = pd.DataFrame([input_data])

        le=LabelEncoder()
        df["Task_Complexity"]=le.fit_transform(df["Task_Complexity"])
        
        expected_cols=[
            'CPU_Usage','Battery_Level','Network_Latency',
            'Bandwidth_Availability','Data_Size','Task_Complexity'
        ]

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0  
        if 'User_ID' in df.columns:
            df.drop('User_ID',axis=1,inplace=True)

        df = df[expected_cols]

        prediction = model_rf.predict(df)[0]
        result = "Offload to Cloud" if prediction == 1 else "Execute Locally"

        messagebox.showinfo("Prediction", f"Decision: {result}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

for field in fields:
    label = tk.Label(root, text=field)
    label.pack()
    entry = tk.Entry(root)
    entry.pack()
    entries[field] = entry

task_complexity_var = tk.StringVar(root)
task_complexity_var.set("medium")
tk.Label(root, text="Task Complexity").pack()
tk.OptionMenu(root, task_complexity_var, "low", "medium", "high").pack()

predict_btn = tk.Button(root, text="Predict Decision", command=predict_decision)
predict_btn.pack(pady=20)

plt.show()
root.mainloop()
root_rule.mainloop()