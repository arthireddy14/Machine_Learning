import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression 

# page config 
st.set_page_config("Multiple Linear Regression ",layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f'<style> {f.read()}</style>',unsafe_allow_html=True)
load_css("styles.css")

# Title
st.markdown("""
<div class="card">
<h1>Multiple Linear Regression</h1>
<p>Predict <b>Tip Amount </b> from <b> Total bill </b> using Linear Regression</p>
</div>
""",unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

# Data preview 
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)

# Prepare data
x,y=df[["total_bill","size"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# Train model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

# Metrics
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)

# Visualization
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.6)
ax.plot(df["total_bill"],model.predict(scaler.transform(x)),color="red")
ax.set_xlabel("Total bill ($)")
ax.set_ylabel("Tip ($)")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

# Performance
st.markdown(f"""
            <div class="card">
            <h3> Model Interception</h3>
            <p><b> Co-efficient(Total_bill):</b> {model.coef_[0]:.3f}<br>
            <b>co-efficient(Group size):</b> {model.coef_[1]:.3f}<br>
            <b>Intercept: </b> {model.intercept_:.3f}</p>
            <p>
            Tip depends upon the <b> Bill amount </b> and <b> Number of people</b>.
            </p>
            </div>
            """,unsafe_allow_html=True)

# Prediction
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Predict tip amount")
min_bill = float(df.total_bill.min())
max_bill = float(df.total_bill.max())
bill = st.slider("Total Bill Amount", min_value=min_bill, max_value=max_bill, value=30.0)

size=st.slider("Group size ",int(df["size"].min()),int(df["size"].max()),2)
input_scaled=scaler.transform([[bill,size]])
st.markdown('<div class="card">',unsafe_allow_html=True)
tip=model.predict(input_scaled)[0]
st.markdown(f'<div class="prediction-box"> Predicted Tip: ${tip:.2f}</div>',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)
