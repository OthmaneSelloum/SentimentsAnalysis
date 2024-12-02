# SentimentsAnalysis
# 📊 Sentiment Analysis Project  

This project implements sentiment analysis on textual data using <b>Deep Learning models</b>, specifically a **Convolutional Neural Network (CNN)** and a **Long Short-Term Memory (LSTM)** network.  

---

## 🌟 Features  

- 📂 **Dataset Handling**:  
  - Loaded and prepared a labeled dataset from IMDB for training and testing. 

- 🔄 **Data Preprocessing**:  
  - 📝 **Tokenization**: Splitting text into tokens.  
  - 🛑 **Stopword Removal**: Removing unnecessary words for better analysis.  
  - ✂️ **Text Cleaning**: Stripping special characters and numbers.  
  - 📏 **Padding**: Standardizing sequence lengths for model compatibility.  

- 💡 **Embeddings**:  
  - Generated numerical representations of words using embedding layers to capture semantic relationships.  

- 🧠 **Model Training**:  
  - Trained CNN and LSTM models on the processed dataset.  

- 📈 **Evaluation**:  
  - Visualized results and evaluated model performance using key metrics.  

---

## 📦 Requirements  

To run the project, you need the following libraries:  

- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`  
- `nltk`  
- `tensorflow`  
- `scikit-learn`  

## 📂 Project Structure
plaintext
Copier le code
Sentiments Analysis/
├── data/                    # Dataset folder
├── Sentiments Analysis.ipynb # Jupyter Notebook
├── models/                  # Saved trained models (optional)
└── README.md                # Project Documentation

### 📋 **Rapport d'Entraînement des Modèles**

Dans ce projet, nous avons entraîné deux modèles, un **Convolutional Neural Network (CNN)** et un **Long Short-Term Memory (LSTM)**, pour effectuer une analyse de sentiments sur notre dataset. Voici un résumé détaillé de nos observations :  

---

#### ⚡ **1. Modèle CNN (Convolutional Neural Network)**  
- **🔧 Architecture** :  
  - Une couche d'**embedding** suivie d'une couche de convolution 1D, d'un **Global Max Pooling** et d'une couche dense pour la sortie.  
- **📈 Performance** :  
  - Pendant l'entraînement, le modèle CNN a obtenu une haute précision sur l'ensemble d'entraînement.  
  - Cependant, nous avons observé un **surapprentissage** significatif : les performances sur l'ensemble de validation ont commencé à se dégrader. Cela montre que le modèle avait du mal à se généraliser à des données non vues, probablement en raison de sa dépendance à des motifs localisés dans les données.  

---

#### 🌊 **2. Modèle LSTM (Long Short-Term Memory)**  
- **🔧 Architecture** :  
  - Une couche d'**embedding**, suivie d'une couche LSTM et d'une couche dense pour la sortie.  
- **📈 Performance** :  
  - Le modèle LSTM a montré une meilleure capacité de **généralisation** par rapport au CNN.  
  - Les performances sur les ensembles d'entraînement et de validation étaient alignées, ce qui indique un **faible risque de surapprentissage**. Grâce à sa capacité à capturer les relations séquentielles et les dépendances à long terme dans les données, il s'est révélé plus efficace.  

---

#### ✅ **Conclusion**  
Après analyse, nous avons choisi de continuer avec le **modèle LSTM** pour nos prédictions.  
👉 Il est mieux adapté à la nature séquentielle du dataset et évite le problème de surapprentissage rencontré avec le modèle CNN. Cette décision garantit de meilleures performances et une meilleure capacité de généralisation aux nouvelles données.  

## **📝 Notes**
This project demonstrates the integration of machine learning with natural language processing for sentiment classification. Further enhancements can include hyperparameter tuning and experimenting with other models.
