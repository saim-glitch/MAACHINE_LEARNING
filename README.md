# 📱 Spam Message Classifier

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.6+-orange.svg)

<p align="center">
  <img src="https://raw.githubusercontent.com/yourusername/spam-classifier/main/demo.gif" alt="App Demo" width="600">
</p>

## 🚀 Try It Now

Visit our [live demo](https://spam-classifier-demo.streamlit.app/) to test the classifier with your own messages!

## ✨ Features

- 🔍 **Powerful Detection**: Accurately identifies spam messages using ensemble learning
- 🔄 **Real-time Processing**: Get instant classification results
- 📊 **Confidence Scores**: See how confident the model is about its prediction
- 🧠 **Ensemble Learning**: Combines SVM, Naive Bayes, and Extra Trees classifiers

## 🛠️ Tech Stack

- **Python 3.x**
- **Scikit-learn**
- **NLTK**
- **Streamlit**
- **Pickle**

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/saim-glitch/MAACHINE_LEARNING.git
cd spam-classifier

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🧪 Model Performance

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 97.8%  |
| Precision   | 95.6%  |
| Recall      | 96.3%  |
| F1 Score    | 96.0%  |

## 📋 Example Classifications

| Message | Classification | Confidence |
|---------|---------------|------------|
| "Congratulations! You've won a free iPhone. Call now to claim!" | ❌ Spam | 98.2% |
| "Hey, can we meet at 5pm tomorrow for coffee?" | ✅ Not Spam | 99.4% |
| "URGENT: Your account has been compromised. Click here to reset your password" | ❌ Spam | 97.8% |

## 📁 Project Structure

```
spam-classifier/
├── app.py                     # Streamlit application
├── vectorizer.pkl             # TF-IDF vectorizer
├── voting_classifier_model.pkl # Trained model
├── requirements.txt           # Dependencies
└── notebooks/                 # Development notebooks
```

## 🔄 How It Works

1. **Text Preprocessing**
   - Tokenization
   - Stopword removal
   - Stemming

2. **Feature Extraction**
   - TF-IDF Vectorization

3. **Ensemble Classification**
   - SVC (kernel='sigmoid')
   - MultinomialNB
   - ExtraTreesClassifier
   - Soft voting combination

## 📈 Future Roadmap

- [ ] Multi-language support
- [ ] API endpoint for integration
- [ ] User feedback loop
- [ ] Mobile app version

## 👥 Contributors

- [Your Name](https://github.com/yourusername)

## 📄 License

MIT © [Your Name]
