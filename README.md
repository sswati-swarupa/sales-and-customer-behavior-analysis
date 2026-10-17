# 🛒 E-commerce Sales & Customer Purchase Behavior Analysis

This project analyzes e-commerce sales performance and customer purchase behavior using Python, pandas, Plotly, and Streamlit.

The dashboard helps identify revenue trends, top-selling products, high-revenue countries, peak sales times, valuable customer segments, and inactive customers who may need retention offers.

## 📊 Features

- Upload UCI Online Retail dataset in Excel or CSV format
- View total revenue, orders, customers, and products
- Analyze monthly revenue trends
- Identify top 10 products by revenue
- Analyze top 10 countries by revenue
- Explore revenue by weekday and hour
- Perform RFM customer segmentation
- Identify inactive customers for retention campaigns
- Optional repeat-customer prediction using Random Forest

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Openpyxl
- Scikit-learn

## 📁 Dataset

This project uses the UCI Online Retail dataset.

- UCI dataset link: https://archive.ics.uci.edu/dataset/352/online+retail
- Kaggle CSV version: https://www.kaggle.com/datasets/carrie1/ecommerce-data

Upload the dataset after opening the Streamlit app.

## 🚀 How to Run Locally

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open the app in your browser:

```text
http://localhost:8501
```

## 📦 requirements.txt

```txt
streamlit
pandas
numpy
plotly
openpyxl
scikit-learn
```

## 📈 Key Insights

- Monthly revenue trends show seasonal changes in customer purchases.
- A small group of products contributes a large share of total revenue.
- The United Kingdom is the main revenue-generating market.
- RFM analysis identifies best, loyal, regular, and at-risk customers.
- Inactive high-value customers should receive priority retention offers.

## 👤 Author

SWATI SWARUPA SAHOO
Python and Data Analysis Project 
