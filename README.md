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
  - Generated numerical representations of words using embedding layers(GloVe) to capture semantic relationships.  

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
- `tensorflow-keras`  
- `scikit-learn`  

## 📂 Project Structure

Sentiments Analysis/
├── data/                    # Dataset folder
├── Sentiments Analysis.ipynb # Jupyter Notebook
├── models/                  # Saved trained models (optional)
└── README.md                # Project Documentation

---

### 📋 **Model Training Report**

In this project, we trained two models, a **Convolutional Neural Network (CNN)** and a **Long Short-Term Memory (LSTM)** network, on our dataset to perform sentiment analysis. Below is a detailed summary of our observations:  

---

#### ⚡ **1. CNN Model (Convolutional Neural Network)**  
- **🔧 Architecture**:  
  - The CNN model featured an **embedding layer**, followed by a 1D convolutional layer, **Global Max Pooling**, and a dense output layer.  
- **📈 Performance**:  
  - The CNN model achieved high accuracy on the training set during training.  
  - However, we observed significant **overfitting**: performance on the validation set degraded as training progressed. This suggests the model struggled to generalize to unseen data, likely due to its reliance on localized patterns in the dataset.  

---

#### 🌊 **2. LSTM Model (Long Short-Term Memory)**  
- **🔧 Architecture**:  
  - The LSTM model included an **embedding layer**, followed by an LSTM layer and a dense output layer.  
- **📈 Performance**:  
  - The LSTM model demonstrated better **generalization** compared to the CNN.  
  - Training and validation performance were closely aligned, indicating a **lower risk of overfitting**. Its ability to capture sequential dependencies and long-term relationships in the data made it more effective for this task.  

---

#### ✅ **Conclusion**  
After careful analysis, we decided to proceed with the **LSTM model** for predictions.  
👉 It better handles the sequential nature of the dataset and avoids the overfitting issues encountered with the CNN model. This choice ensures better performance and generalization to unseen data.  

----

## **📝 Notes**
This project demonstrates the integration of machine learning with natural language processing for sentiment classification. Further enhancements can include hyperparameter tuning and experimenting with other models.
