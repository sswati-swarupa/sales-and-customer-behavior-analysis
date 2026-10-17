import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Page settings
st.set_page_config(
    page_title="E-commerce Sales & Customer Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-commerce Sales & Customer Purchase Behavior Analysis")
st.markdown(
    """
    Upload the **UCI Online Retail dataset** (Excel or CSV) to analyze sales performance,
    product performance, country-wise revenue, customer behavior, and RFM segments.
    """
)

# ------------------------------
# Data loading and cleaning
# ------------------------------

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(uploaded_file)

    # Basic cleaning
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.drop_duplicates()

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Feature engineering
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["MonthName"] = df["InvoiceDate"].dt.strftime("%b")
    df["Weekday"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    return df


@st.cache_data
def create_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()

    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    )
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    def segment_customer(score):
        if score >= "444":
            return "Best Customers"
        elif score.startswith("4") or score.startswith("5"):
            return "Loyal Customers"
        elif score.startswith("1"):
            return "At-Risk Customers"
        else:
            return "Regular Customers"

    rfm["Segment"] = rfm["RFM_Score"].apply(segment_customer)

    return rfm


# ------------------------------
# File upload
# ------------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    rfm = create_rfm(df)

    # --------------------------
    # Sidebar filters
    # --------------------------
    st.sidebar.header("🔎 Filters")

    countries = st.sidebar.multiselect(
        "Select Country",
        options=sorted(df["Country"].unique()),
        default=sorted(df["Country"].unique())
    )

    months = st.sidebar.multiselect(
        "Select Month",
        options=sorted(df["MonthName"].unique()),
        default=sorted(df["MonthName"].unique())
    )

    filtered_df = df[
        (df["Country"].isin(countries)) &
        (df["MonthName"].isin(months))
    ]

    # --------------------------
    # KPI cards
    # --------------------------
    total_revenue = filtered_df["Revenue"].sum()
    total_orders = filtered_df["InvoiceNo"].nunique()
    total_customers = filtered_df["CustomerID"].nunique()
    total_products = filtered_df["StockCode"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("🧾 Total Orders", f"{total_orders:,}")
    col3.metric("👥 Total Customers", f"{total_customers:,}")
    col4.metric("📦 Total Products", f"{total_products:,}")

    st.markdown("---")

    # --------------------------
    # Monthly revenue trend
    # --------------------------
    st.subheader("📈 Monthly Revenue Trend")

    monthly_revenue = (
        filtered_df.groupby(["Year", "Month", "MonthName"], as_index=False)["Revenue"]
        .sum()
        .sort_values(["Year", "Month"])
    )

    fig_month = px.line(
        monthly_revenue,
        x="MonthName",
        y="Revenue",
        color="Year",
        markers=True,
        title="Revenue by Month"
    )

    st.plotly_chart(fig_month, use_container_width=True)

    best_month = monthly_revenue.sort_values("Revenue", ascending=False).iloc[0]
    st.success(
        f"**Finding:** {best_month['MonthName']} {best_month['Year']} generated the highest revenue "
        f"of ₹{best_month['Revenue']:,.0f}. This may be due to seasonal demand, promotions, or bulk purchases."
    )

    # --------------------------
    # Top products
    # --------------------------
    st.subheader("🏆 Top 10 Products by Revenue")

    top_products = (
        filtered_df.groupby("Description", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )

    fig_products = px.bar(
        top_products.sort_values("Revenue"),
        x="Revenue",
        y="Description",
        orientation="h",
        title="Top 10 Products by Revenue",
        color="Revenue",
        color_continuous_scale="Teal"
    )

    st.plotly_chart(fig_products, use_container_width=True)

    best_product = top_products.iloc[0]
    st.info(
        f"**Finding:** '{best_product['Description']}' generated the highest revenue "
        f"of ₹{best_product['Revenue']:,.0f}. The store should maintain stock and promote this product."
    )

    # --------------------------
    # Country-wise revenue
    # --------------------------
    st.subheader("🌍 Top 10 Countries by Revenue")

    country_revenue = (
        filtered_df.groupby("Country", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )

    fig_country = px.bar(
        country_revenue,
        x="Country",
        y="Revenue",
        title="Top 10 Countries by Revenue",
        color="Revenue",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig_country, use_container_width=True)

    best_country = country_revenue.iloc[0]
    st.info(
        f"**Finding:** {best_country['Country']} generated the highest revenue "
        f"of ₹{best_country['Revenue']:,.0f}. It is the most important market for the business."
    )

    # --------------------------
    # Weekday and hourly sales
    # --------------------------
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("📅 Revenue by Weekday")

        weekday_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]

        weekday_revenue = (
            filtered_df.groupby("Weekday", as_index=False)["Revenue"]
            .sum()
        )

        weekday_revenue["Weekday"] = pd.Categorical(
            weekday_revenue["Weekday"],
            categories=weekday_order,
            ordered=True
        )
        weekday_revenue = weekday_revenue.sort_values("Weekday")

        fig_weekday = px.bar(
            weekday_revenue,
            x="Weekday",
            y="Revenue",
            title="Revenue by Weekday",
            color="Revenue",
            color_continuous_scale="Oranges"
        )

        st.plotly_chart(fig_weekday, use_container_width=True)

    with col6:
        st.subheader("⏰ Revenue by Hour")

        hourly_revenue = (
            filtered_df.groupby("Hour", as_index=False)["Revenue"]
            .sum()
            .sort_values("Hour")
        )

        fig_hour = px.line(
            hourly_revenue,
            x="Hour",
            y="Revenue",
            markers=True,
            title="Revenue by Hour of Day"
        )

        st.plotly_chart(fig_hour, use_container_width=True)

    best_weekday = weekday_revenue.sort_values("Revenue", ascending=False).iloc[0]
    best_hour = hourly_revenue.sort_values("Revenue", ascending=False).iloc[0]

    st.success(
        f"**Finding:** {best_weekday['Weekday']} had the highest revenue, and "
        f"{best_hour['Hour']}:00 was the peak sales hour. Marketing campaigns and discounts "
        f"should be scheduled during these periods."
    )

    # --------------------------
    # RFM customer analysis
    # --------------------------
    st.markdown("---")
    st.subheader("👥 RFM Customer Segmentation")

    segment_summary = rfm.groupby("Segment").agg(
        Total_Revenue=("Monetary", "sum"),
        Customer_Count=("CustomerID", "count"),
        Average_Spend=("Monetary", "mean")
    ).sort_values("Total_Revenue", ascending=False).reset_index()

    col7, col8 = st.columns(2)

    with col7:
        fig_segment_revenue = px.pie(
            segment_summary,
            names="Segment",
            values="Total_Revenue",
            title="Revenue Contribution by Customer Segment",
            hole=0.4
        )
        st.plotly_chart(fig_segment_revenue, use_container_width=True)

    with col8:
        fig_segment_count = px.bar(
            segment_summary,
            x="Segment",
            y="Customer_Count",
            title="Number of Customers by Segment",
            color="Customer_Count",
            color_continuous_scale="Purples"
        )
        st.plotly_chart(fig_segment_count, use_container_width=True)

    best_segment = segment_summary.iloc[0]
    st.success(
        f"**Finding:** '{best_segment['Segment']}' is the most valuable segment, generating "
        f"₹{best_segment['Total_Revenue']:,.0f} from {best_segment['Customer_Count']:,} customers. "
        f"The business should retain these customers through loyalty rewards and personalized offers."
    )

    # --------------------------
    # Inactive customers
    # --------------------------
    st.subheader("⚠️ Customers Needing Retention Offers")

    inactive_days = st.slider(
        "Select inactive days",
        min_value=30,
        max_value=365,
        value=90
    )

    inactive_customers = rfm[rfm["Recency"] > inactive_days].sort_values(
        "Monetary", ascending=False
    )

    st.write(
        f"**{len(inactive_customers):,} customers** have not purchased in the last "
        f"{inactive_days} days."
    )

    st.dataframe(
        inactive_customers[
            ["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]
        ].head(20)
    )

    st.warning(
        "High-value inactive customers should receive priority win-back emails, discount offers, "
        "or personalized product recommendations."
    )

    # --------------------------
    # Optional repeat-purchase model
    # --------------------------
    st.markdown("---")
    st.subheader("🤖 Optional: Repeat Customer Prediction")

    if st.button("Train Repeat Customer Model"):
        first_purchase = df.groupby("CustomerID")["InvoiceDate"].min()
        last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()

        customer_df = pd.DataFrame({
            "FirstPurchase": first_purchase,
            "LastPurchase": last_purchase
        })

        customer_df["RepeatCustomer"] = (
            customer_df["FirstPurchase"] != customer_df["LastPurchase"]
        ).astype(int)

        model_data = rfm.merge(customer_df, on="CustomerID")

        X = model_data[["Recency", "Frequency", "Monetary"]]
        y = model_data["RepeatCustomer"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
        st.text("Classification Report")
        st.code(classification_report(y_test, y_pred))

else:
    st.warning("Please upload the UCI Online Retail dataset in `.xlsx` or `.csv` format.")
    st.markdown(
        """
        **Dataset options:**
        - UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail
        - Kaggle CSV version: https://www.kaggle.com/datasets/carrie1/ecommerce-data
        """
    )