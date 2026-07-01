readme_code = '''# ✍️ Handwritten Digit Recognition using Deep Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-7c3aed?style=flat-square&logoColor=white)](LICENSE)

A deep learning web application that recognizes handwritten digits (0–9) in real time, built with an Artificial Neural Network (ANN) trained on the MNIST dataset and deployed as an interactive Streamlit app.

---

## 🔗 Live Demo

**👉 [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)**

---

## 📖 Project Overview

This project implements an end-to-end machine learning pipeline for handwritten digit recognition:

1. **Model Development** — an ANN trained on the MNIST dataset (60,000 training images, 10,000 test images)
2. **Prediction Pipeline** — a reusable module (`predict.py`) that preprocesses any digit image and returns a prediction
3. **Web Application** — an interactive Streamlit app (`app.py`) where users draw a digit on a canvas and get an instant prediction with confidence score

---

## ✨ Features

- 🎨 Interactive drawing canvas — draw any digit with your mouse
- 🔮 Real-time digit prediction with confidence percentage
- 📊 Probability distribution bar chart across all 10 digit classes
- 🧹 One-click clear/reset
- 📱 Clean, responsive Streamlit UI with sidebar instructions

---

## 📊 Dataset

**MNIST Handwritten Digits Dataset**

| Attribute | Value |
|:----------|:------|
| Training images | 60,000 |
| Test images | 10,000 |
| Image size | 28 × 28 pixels, grayscale |
| Classes | 10 (digits 0–9) |
| Source | `tensorflow.keras.datasets.mnist` |

---

## 🧠 Model Architecture