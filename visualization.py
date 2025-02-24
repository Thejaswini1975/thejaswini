import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Sample Data
text_quality = {"Completeness": 60, "Noise": 15, "Ambiguity": 15, "Redundancy": 10}
text_anomalies = {"Spelling Errors": 30, "Grammar Issues": 25, "Duplicate Entries": 20, "Gibberish": 25}
word_counts = Counter({"AI": 10, "NLP": 7, "Data": 15, "Quality": 5, "Analysis": 8})

# 1️⃣ Gauge Chart for Text Quality Score
def plot_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Text Quality Score"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "green"},
               'steps': [
                   {'range': [0, 50], 'color': "red"},
                   {'range': [50, 80], 'color': "yellow"},
                   {'range': [80, 100], 'color': "green"}]
              }
    ))
    fig.show()

# 2️⃣ Doughnut Chart for Text Quality Breakdown
def plot_text_quality_pie():
    labels = list(text_quality.keys())
    sizes = list(text_quality.values())
    colors = ["green", "orange", "blue", "red"]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'})
    plt.title("Text Quality Breakdown")
    plt.show()

# 3️⃣ Bar Chart for Anomaly Detection
def plot_anomaly_bar_chart():
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(text_anomalies.keys()), y=list(text_anomalies.values()), palette='Reds_r')
    plt.xlabel("Anomaly Type")
    plt.ylabel("Count")
    plt.title("Text Anomaly Detection")
    plt.xticks(rotation=30)
    plt.show()

# 4️⃣ Word Frequency Bar Chart
def plot_word_frequency():
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(word_counts.keys()), y=list(word_counts.values()), palette='Blues_r')
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Word Frequency Analysis")
    plt.show()

# 5️⃣ ROC Curve
def plot_roc_curve():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])  # Example true labels
    y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.85, 0.15, 0.75, 0.05])  # Example scores
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

# 6️⃣ Confusion Matrix
def plot_confusion_matrix():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])  # Example true labels
    y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 0])  # Example predicted labels
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# 7️⃣ Heatmap for Confusion Matrix
def plot_heatmap_confusion_matrix():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 0])
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

# Generate Visualizations
plot_gauge_chart(75)  # Example text quality score
plot_text_quality_pie()
plot_anomaly_bar_chart()
plot_word_frequency()
plot_roc_curve()
plot_confusion_matrix()
plot_heatmap_confusion_matrix()
