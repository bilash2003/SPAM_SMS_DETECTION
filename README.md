📩 SMS Spam Detection using Machine Learning & BERT

A complete end-to-end SMS Spam Detection System built using
. Natural Language Processing (NLP)
. Machine Learning algorithms (Naïve Bayes, SVM, Logistic Regression, etc.)
. TF-IDF Vectorization
. BERT (Transformer-based deep learning model)
. Streamlit for deployment

This project classifies a given SMS message as SPAM or HAM.
🚀 Features
✔ End-to-end Data Preprocessing
✔ NLP Pipeline (Tokenization → Stopword Removal → Stemming)
✔ ML Models Comparison
✔ TF-IDF Vectorizer
✔ BERT-based classifier using Transformers
✔ Visualizations (WordCloud, Histograms, Pairplots)
✔ Exported Model (model.pkl, vectorizer.pkl)
✔ Streamlit deployment support

📂 Project Structure
📁 sms-spam-detection
│── app.py
│── spam.csv
│── model.pkl
│── vectorizer.pkl
│── requirements.txt
│── nltk.txt
│── setup.sh
│── Procfile
│── spam_detection.ipynb
│── README.md (you will paste this file)

📊 Dataset Information
Dataset used: SMS Spam Collection Dataset
Source: Contains ~5,500 SMS labelled as "ham" or "spam".
Columns used:
v1 → target (ham/spam)
v2 → text (SMS text)
Unnecessary columns removed:
Unnamed: 2
Unnamed: 3
Unnamed: 4

🧹 Data Preprocessing & NLP Pipeline

✔ Step 1 — Cleaning & Renaming Columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
df.rename(columns={'v1':'target','v2':'text'}, inplace=True)

✔ Step 2 — Label Encoding
ham → 0  
spam → 1

✔ Step 3 — Remove Duplicates
✔ Step 4 — Feature Engineering

Columns created:
num_characters
num_words
num_sentences

✔ Step 5 — Text Transformation Function
Your pipeline:
Lowercase
Tokenization
Remove special characters
Remove stopwords
Remove punctuation

Apply Porter Stemming

🌐 ML Vectorization Techniques Used
✔ CountVectorizer

Transforms text → numerical bag-of-words vectors.

✔ TF-IDF Vectorizer

Used with max_features=3000
This vectorizer gave the best precision.

🤖 Machine Learning Models Used

You trained & compared the following:

Model	Accuracy	Precision: 
Multinomial Naive Bayes	⭐ Best	⭐ Best

SVM (Sigmoid Kernel)	Good	High

Logistic Regression	Good	Good

KNN	Moderate	Low

Decision Tree	Moderate	Low

Random Forest	High	Good

AdaBoost	Good	Good

You also built:
✔ Voting Classifier (Soft Voting)

Models:

SVM
MultinomialNB
ExtraTreesClassifier
✔ Stacking Classifier
Meta model: RandomForestClassifier
🤗 BERT / Transformers Model

You fine-tuned:
distilbert-base-uncased
bert-base-uncased

Steps:
Tokenization
Transform into input_ids & attention_mask
Train/test split
Training with Trainer API
Evaluate metrics
Prediction function:

def predict_sms(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = softmax(outputs.logits)
    return "SPAM" or "HAM"

📈 Visualizations Included

Histogram (ham vs spam)

WordCloud for spam

Spam frequent words barplot

Pairplot for feature correlation

🧪 Model Files

The following files are used for deployment:

model.pkl          → trained ML model  
vectorizer.pkl     → TF-IDF vectorizer  
app.py             → streamlit app  

▶️ How to Run Locally
1. Clone the repo
git clone https://github.com/yourusername/sms-spam-detection.git
cd sms-spam-detection

2. Install dependencies
pip install -r requirements.txt

3. Run Streamlit App
streamlit run app.py

🚀 Deployment (Streamlit / Render / Heroku)

Your repo already has:

✔ Procfile
✔ setup.sh
✔ requirements.txt
✔ nltk.txt

This means your app can be deployed on:

Streamlit Cloud

Push to GitHub

Go to share.streamlit.io

Select repo → select app.py → Deploy

Render / Railway

Supported via Procfile.

🛠 Technologies Used
Programming Language: Python 3.x 

Libraries
Data Handling: numpy  pandas

Visualization: matplotlib seaborn wordcloud

NLP:  nltk stopwords PorterStemmer

ML Models: scikit-learn CountVectorizer TfidfVectorizer

Deep Learning: transformers torch datasets

Deployment: streamlit

📌 Future Improvements
Add LSTM model
Add FastAPI backend
Add continuous monitoring
Improve model interpretability (SHAP)
Add UI enhancements

👤 Author

Bilash Mallick
B.Tech CSE (AI & ML)
