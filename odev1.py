import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def show(param1,param2, param3, param4) -> None:
    print(param1,"\n",param2,"\n", param3,"\n", param4)

def main():

    data = pd.read_csv('./data.csv')
    mt = data.shape
    clmn = data.columns
    kontrol = data.isnull().sum()
    show(data.head(),mt,clmn,kontrol)
    
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    model = RandomForestClassifier(random_state= 42)
    model.fit(X_train,y_train)
    
    #TAHMİN
    y_pred = model.predict(X_test)
    
    #METRİKLER
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    show(accuracy,precision,recall,f1)

    #GÖRSELLEŞTİRME
    print("\n--- KARISIKLIK MATRİSİ (CONFUSION MATRIX) ---")
    print(confusion_matrix(y_test, y_pred))
    print("\n--- SINIFLANDIRMA RAPORU ---")
    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)']))


    cm_lr = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Benign (İyi)', 'Malignant (Kötü)'], 
                yticklabels=['Benign (İyi)', 'Malignant (Kötü)'])
    
    plt.title('Logistic Regression - Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.tight_layout()
    
    # Resmi Kaydet
    plt.savefig('grafik-1.png')
    print("\nKarmaşıklık matrisi 'grafik-1.png' olarak başarıyla oluşturuldu.")

    plt.show()
    

if __name__ == "__main__":
    main()



