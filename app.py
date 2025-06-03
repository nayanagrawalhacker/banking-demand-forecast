import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor # Using RF for XGBoost placeholder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Removed: from statsmodels.tsa.arima.model import ARIMA
import warnings
import base64 # For encoding images/files for download

# Try to import econml, but continue if not available
try:
    import econml
    from econml.dml import LinearDML, CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    # Create placeholder classes for econml components
    class LinearDML:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            pass
        def effect(self, *args, **kwargs):
            return np.zeros(1)
        def effect_interval(self, *args, **kwargs):
            return np.zeros(1), np.zeros(1)
    
    class CausalForestDML(LinearDML):
        pass

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Page Configuration (Set initial theme based on session state or default to light) ---
if "theme" not in st.session_state:
    st.session_state.theme = "light" # Default theme

st.set_page_config(
    page_title="Causal AI Banking Demand Forecaster",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
    # Theme is controlled dynamically below
)

# --- Custom CSS for Styling and Theme Control ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Define CSS styles (can be moved to a separate css file if extensive)
light_theme_css = """
:root {
    --primary-color: #0068C9; /* Professional Blue */
    --background-color: #FFFFFF;
    --secondary-background-color: #F0F2F6;
    --text-color: #31333F;
    --secondary-text-color: #555;
    --sidebar-background: #F8F9FA;
    --widget-background: #FFFFFF;
    --border-color: #E6EAF1;
    --hover-background-color: #E6F0FF;
}
body { background-color: var(--background-color); color: var(--text-color); }
.stApp { background-color: var(--background-color); }
.stSidebar { background-color: var(--sidebar-background); }
.stButton>button { 
    border: 1px solid var(--primary-color);
    background-color: var(--primary-color);
    color: white;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background-color: white;
    color: var(--primary-color);
    border-color: var(--primary-color);
}
.stMetric { 
    background-color: var(--secondary-background-color);
    border-radius: 8px;
    padding: 15px;
    border: 1px solid var(--border-color);
}
.stTabs [data-baseweb="tab-list"] { 
    background-color: var(--secondary-background-color);
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"] { 
    background-color: transparent; 
    border-bottom: 2px solid transparent;
    transition: border-color 0.3s ease-in-out;
}
.stTabs [aria-selected="true"] { 
    border-bottom: 2px solid var(--primary-color);
    color: var(--primary-color);
}
/* Add subtle transition for elements appearing/changing */
.stDataFrame, .stPlotlyChart, .stMetric, .stAlert {
    transition: opacity 0.5s ease-in-out;
}
"""

dark_theme_css = """
:root {
    --primary-color: #1F8AFF; /* Brighter Blue for Dark Mode */
    --background-color: #0E1117;
    --secondary-background-color: #1C2026;
    --text-color: #FAFAFA;
    --secondary-text-color: #A0A0A0;
    --sidebar-background: #161A1F;
    --widget-background: #262730;
    --border-color: #303338;
    --hover-background-color: #2C3036;
}
body { background-color: var(--background-color); color: var(--text-color); }
.stApp { background-color: var(--background-color); }
.stSidebar { background-color: var(--sidebar-background); }
.stButton>button { 
    border: 1px solid var(--primary-color);
    background-color: var(--primary-color);
    color: white;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background-color: var(--secondary-background-color);
    color: var(--primary-color);
    border-color: var(--primary-color);
}
.stMetric { 
    background-color: var(--secondary-background-color);
    border-radius: 8px;
    padding: 15px;
    border: 1px solid var(--border-color);
}
.stTabs [data-baseweb="tab-list"] { 
    background-color: var(--secondary-background-color);
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"] { 
    background-color: transparent; 
    border-bottom: 2px solid transparent;
    transition: border-color 0.3s ease-in-out;
}
.stTabs [aria-selected="true"] { 
    border-bottom: 2px solid var(--primary-color);
    color: var(--primary-color);
}
/* Add subtle transition for elements appearing/changing */
.stDataFrame, .stPlotlyChart, .stMetric, .stAlert {
    transition: opacity 0.5s ease-in-out;
}
"""

# Apply the selected theme CSS
st.markdown(f"<style>{dark_theme_css if st.session_state.theme == 'dark' else light_theme_css}</style>", unsafe_allow_html=True)

# --- Helper Functions ---

# Function to encode image for embedding (if needed, e.g., flowchart)
# def get_image_base64(path):
#     with open(path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()

# Function to create download link for dataframes
def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # some strings <-> bytes conversions necessary here
    # Style the link like a button
    button_style = """
    display: inline-block;
    padding: 0.5em 1em;
    color: white;
    background-color: var(--primary-color);
    border: none;
    border-radius: 5px;
    text-decoration: none;
    text-align: center;
    cursor: pointer;
    font-size: 14px;
    margin-top: 10px;
    transition: background-color 0.3s ease;
    """
    hover_style = "background-color: #0056b3;" # Darker blue on hover for light theme
    if st.session_state.theme == "dark":
        hover_style = "background-color: #1A75D1;" # Lighter blue on hover for dark theme
        
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="{button_style}" onmouseover="this.style.cssText+=\"{hover_style}\"" onmouseout="this.style.cssText=\"{button_style}\"">{text}</a>'
    return href

# --- Data Loading and Caching ---
@st.cache_data # Use Streamlit's caching to load data only once
def load_data(file_path):
    """Loads the banking demand dataset and performs initial preprocessing."""
    try:
        df = pd.read_csv(file_path)
        df["Month"] = pd.to_datetime(df["Month"] + "-01")
        df = df.sort_values("Month")
        num_cols = ["Product_Demand", "Ad_Spend", "Interest_Rate", "Market_Index", "Previous_Month_Demand"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=num_cols, inplace=True)
        df["Year"] = df["Month"].dt.year
        df["MonthNum"] = df["Month"].dt.month
        df["Month_Since_Start"] = (df["Year"] - df["Year"].min()) * 12 + df["MonthNum"]
        # Add lagged features needed for some models
        df["Demand_Lag1"] = df.groupby(["Product", "Region", "Customer_Segment"])["Product_Demand"].shift(1)
        df.dropna(subset=["Demand_Lag1"], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# Load the data
data_path = "simulated_banking_demand_dataset.csv"
df = load_data(data_path)

# --- Model Training Functions (Cached) ---
@st.cache_data
def preprocess_for_ml(_df_ml, include_causal_features=False, causal_features_df=None, ts_features=None):
    """Prepares data for ML models, optionally including causal features and specific TS features."""
    df_processed = _df_ml.copy()
    
    # Add causal features if provided
    if include_causal_features and causal_features_df is not None:
        # Ensure index alignment before merging
        causal_features_df = causal_features_df.reindex(df_processed.index)
        df_processed = pd.merge(df_processed, causal_features_df, left_index=True, right_index=True, how='left')
        # Fill potential NaNs introduced by merge or calculation (e.g., with 0 or mean)
        for col in causal_features_df.columns:
             if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0) 

    # One-hot encode categorical features
    categorical_features = ["Product", "Region", "Customer_Segment"]
    df_encoded = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True)
    
    # Define features (X) and target (y)
    exclude_cols = ["Product_Demand", "Month"] # Exclude target and original date
    
    if ts_features: # Use specific features for simple TS model
        features = [f for f in ts_features if f in df_encoded.columns]
    else: # Use all other features for general ML models
        features = [col for col in df_encoded.columns if col not in exclude_cols]
        
    X = df_encoded[features]
    y = df_encoded["Product_Demand"]
    
    # Split data (using time-based split for forecasting)
    split_point = int(len(df_encoded) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    test_dates = df_processed["Month"][split_point:]
    
    return X_train, X_test, y_train, y_test, test_dates, features

@st.cache_resource
def train_evaluate_linear_regression(_df_filtered):
    """Trains and evaluates a Linear Regression model."""
    X_train, X_test, y_train, y_test, test_dates, _ = preprocess_for_ml(_df_filtered.copy())
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)
    results_df = pd.DataFrame({"Month": test_dates, "Actual": y_test, "Predicted": predictions})
    return model, rmse, mape, results_df

@st.cache_resource
def train_evaluate_xgboost(_df_filtered):
    """Trains and evaluates an XGBoost model (using RandomForest as placeholder)."""
    X_train, X_test, y_train, y_test, test_dates, _ = preprocess_for_ml(_df_filtered.copy())
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)
    results_df = pd.DataFrame({"Month": test_dates, "Actual": y_test, "Predicted": predictions})
    return model, rmse, mape, results_df

# Removed: train_evaluate_arima function

@st.cache_resource
def train_evaluate_simple_ts(_df_filtered):
    """Trains and evaluates a simple time-series model using lagged features."""
    # Use only lagged demand and time features
    ts_features_to_use = ["Demand_Lag1", "MonthNum", "Year", "Month_Since_Start"]
    X_train, X_test, y_train, y_test, test_dates, _ = preprocess_for_ml(_df_filtered.copy(), ts_features=ts_features_to_use)
    
    if X_train.empty or X_test.empty:
        return None, np.nan, np.nan, pd.DataFrame()
        
    # Use Linear Regression for simplicity
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)
    results_df = pd.DataFrame({"Month": test_dates, "Actual": y_test, "Predicted": predictions})
    return model, rmse, mape, results_df

@st.cache_data
def get_confounders(_df_encoded, _treatment_var, _outcome_var):
    """Identifies confounder columns."""
    exclude_cols = [_outcome_var, _treatment_var, "Month", "Year", "MonthNum"]
    confounders = [col for col in _df_encoded.columns if col not in exclude_cols and _df_encoded[col].dtype in [np.int64, np.float64, np.uint8]]
    return confounders[:15] # Limit confounders

@st.cache_resource
def train_causal_model(_df_causal, _treatment_var, _outcome_var):
    """Trains a causal model using LinearDML and returns the model and effects."""
    if not ECONML_AVAILABLE:
        # st.warning("Causal inference features are disabled in this deployment due to dependency constraints.")
        return None, np.nan, np.nan, np.nan, None, None
        
    try:
        categorical_features = ["Product", "Region", "Customer_Segment"]
        df_encoded = pd.get_dummies(_df_causal, columns=categorical_features, drop_first=True)
        confounders = get_confounders(df_encoded, _treatment_var, _outcome_var)
        if not confounders:
             # st.warning(f"No suitable confounders found for causal analysis of {_treatment_var}.")
             return None, np.nan, np.nan, np.nan, None, None

        X = df_encoded[confounders]
        T = df_encoded[_treatment_var]
        Y = df_encoded[_outcome_var]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=confounders, index=X.index)
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X_scaled_df, T, Y, test_size=0.3, random_state=42)

        model = LinearDML(
            model_y=RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42), # Simplified models
            model_t=RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
            discrete_treatment=(_treatment_var == 'Promotion_Active'),
            cv=2 # Further reduced CV folds
        )
        model.fit(Y_train, T_train, X=X_train)
        
        ate = model.effect(X_test).mean()
        lower_ci, upper_ci = model.effect_interval(X_test, alpha=0.05)
        avg_lower_ci = lower_ci.mean()
        avg_upper_ci = upper_ci.mean()
        
        # Calculate effects for the entire dataset (used for feature creation)
        effects_on_all_data = model.effect(X_scaled_df)
        effects_df = pd.DataFrame({f"effect_{_treatment_var}": effects_on_all_data}, index=X_scaled_df.index)
        
        return model, ate, avg_lower_ci, avg_upper_ci, X_test, effects_df

    except Exception as e:
        # st.error(f"Causal model training failed for {_treatment_var}: {e}")
        return None, np.nan, np.nan, np.nan, None, None

@st.cache_resource
def train_evaluate_causal_ml(_df_filtered):
    """Trains an ML model incorporating causal features."""
    if not ECONML_AVAILABLE:
        st.warning("Causal ML model is using standard RandomForest without causal features due to dependency constraints.")
        # Fall back to standard XGBoost/RandomForest if econml is not available
        return train_evaluate_xgboost(_df_filtered)
        
    df_causal_ml = _df_filtered.copy()
    outcome_var = "Product_Demand"
    treatment_vars = ["Ad_Spend", "Interest_Rate", "Promotion_Active"]
    causal_features_list = []

    # Estimate effects for each treatment and collect them
    for treat_var in treatment_vars:
        # Use a spinner for user feedback during potentially long calculations
        # with st.spinner(f"Calculating causal effect feature for {treat_var}..."):
        _, _, _, _, _, effects_df = train_causal_model(df_causal_ml, treat_var, outcome_var)
        if effects_df is not None:
            causal_features_list.append(effects_df)
    
    if not causal_features_list:
        st.warning("Could not generate any causal features. Proceeding without them.")
        causal_features_combined = None
    else:
        # Combine causal features (handle potential index mismatches if any)
        causal_features_combined = pd.concat(causal_features_list, axis=1)

    # Preprocess data including causal features
    X_train, X_test, y_train, y_test, test_dates, _ = preprocess_for_ml(
        df_causal_ml, 
        include_causal_features=True, 
        causal_features_df=causal_features_combined
    )
    
    # Train the final ML model (e.g., RandomForest)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)
    results_df = pd.DataFrame({"Month": test_dates, "Actual": y_test, "Predicted": predictions})
    return model, rmse, mape, results_df

# --- Sidebar Navigation & Filters ---
st.sidebar.image("https://img.icons8.com/fluency/96/000000/bank-building.png", width=80) # Placeholder icon
st.sidebar.title("🏦 Banking Demand Forecaster")
st.sidebar.markdown("Navigation")

# Use st.session_state to manage the current page
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home / Overview"

pages = {
    "🏠 Home / Overview": "home",
    "📊 Dataset Explorer & EDA": "eda",
    "📈 Forecasting & Comparison": "forecast",
    "📥 Download & Links": "links"
}

def set_page(page_name):
    st.session_state.page = page_name

for page_name in pages.keys():
    st.sidebar.button(page_name, on_click=set_page, args=(page_name,), use_container_width=True, type="primary" if st.session_state.page == page_name else "secondary")

st.sidebar.divider()
st.sidebar.header("Global Filters")
if df is not None:
    products = sorted(df["Product"].unique())
    regions = sorted(df["Region"].unique())
    segments = sorted(df["Customer_Segment"].unique())
    
    with st.sidebar.expander("Filter Data", expanded=True):
        selected_products = st.multiselect("Products", products, default=products)
        selected_regions = st.multiselect("Regions", regions, default=regions)
        selected_segments = st.multiselect("Customer Segments", segments, default=segments)
        
        if not selected_products: selected_products = products
        if not selected_regions: selected_regions = regions
        if not selected_segments: selected_segments = segments
        
        df_filtered = df[df["Product"].isin(selected_products) & df["Region"].isin(selected_regions) & df["Customer_Segment"].isin(selected_segments)].copy()
        st.caption(f"Showing data for {len(df_filtered)} records.")
else:
    df_filtered = None
    st.sidebar.warning("Data not loaded.")

# --- Theme Toggle ---
st.sidebar.divider()
st.sidebar.header("Appearance")
theme_options = ["Light", "Dark"]
def change_theme():
    st.session_state.theme = st.session_state.theme_radio.lower()

st.sidebar.radio(
    "Select Theme", 
    theme_options, 
    index=theme_options.index(st.session_state.theme.capitalize()), 
    key="theme_radio",
    on_change=change_theme
)

# --- Plotly Template Selection based on Theme ---
plotly_template = "plotly_white" if st.session_state.theme == "light" else "plotly_dark"

# --- Page Content Rendering ---

# --- Home / Overview Page ---
if st.session_state.page == "🏠 Home / Overview":
    st.title("🏦 Causal AI-Based Demand Forecasting for Banking Products")
    st.markdown("""
    Welcome to the interactive application demonstrating **Causal AI-Based Demand Forecasting** for banking products.
    This project explores demand patterns using a simulated dataset and leverages causal inference techniques 
    alongside machine learning to provide robust forecasts and actionable insights into what drives demand.
    """, unsafe_allow_html=True)
    
    # Placeholder for flowchart image - replace with actual path if available
    # try:
    #     st.image("path/to/your/flowchart.png", caption="Project Execution Flowchart")
    # except FileNotFoundError:
    #     st.info("Project flowchart image placeholder.")
        
    st.subheader("Project Goal & Benefits")
    st.markdown("""
    *   **Goal:** To accurately forecast demand for banking products (Loans, SIPs, Credit Cards, etc.) while understanding the causal drivers.
    *   **Benefits:** Enables better resource allocation, targeted marketing campaigns, and informed strategic decisions by quantifying the impact of factors like advertising spend and interest rates.
    """)
    
    st.divider()
    st.subheader("Application Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📊 Data Exploration**")
        st.caption("Visualize trends and distributions.")
    with col2:
        st.markdown("**📈 Interactive Forecasting**")
        st.caption("Compare different models.")
    with col3:
        st.markdown("**🧠 Causal Insights**")
        st.caption("Run 'What-If' scenarios.")
        
    st.divider()
    st.markdown("_Use the sidebar to navigate through the different sections of the application._")

# --- Dataset Explorer & EDA Page ---
elif st.session_state.page == "📊 Dataset Explorer & EDA":
    st.title("📊 Dataset Explorer & Exploratory Data Analysis (EDA)")
    st.markdown("Explore the simulated dataset used for this analysis. Apply global filters using the sidebar.")
    
    if df_filtered is not None and not df_filtered.empty:
        with st.container(): # Use container for better spacing/styling
            st.subheader("Filtered Data Preview")
            st.dataframe(df_filtered)
            st.markdown(get_table_download_link(df_filtered, filename="filtered_data.csv", text="Download Filtered Data (CSV)"), unsafe_allow_html=True)
        
        st.divider()
        st.subheader("Key Variable Distributions")
        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            fig_demand_hist = px.histogram(df_filtered, x="Product_Demand", title="Distribution of Product Demand", nbins=50, template=plotly_template)
            st.plotly_chart(fig_demand_hist, use_container_width=True)
            fig_ad_spend_hist = px.histogram(df_filtered, x="Ad_Spend", title="Distribution of Ad Spend", nbins=50, template=plotly_template)
            st.plotly_chart(fig_ad_spend_hist, use_container_width=True)
        with eda_col2:
            fig_interest_hist = px.histogram(df_filtered, x="Interest_Rate", title="Distribution of Interest Rate", nbins=50, template=plotly_template)
            st.plotly_chart(fig_interest_hist, use_container_width=True)
            fig_market_hist = px.histogram(df_filtered, x="Market_Index", title="Distribution of Market Index", nbins=50, template=plotly_template)
            st.plotly_chart(fig_market_hist, use_container_width=True)
            
        st.divider()
        st.subheader("Demand Trends & Patterns")
        tab1, tab2, tab3 = st.tabs(["Demand Over Time", "Demand by Category", "Correlation Analysis"])
        
        with tab1:
            st.markdown("**Total Monthly Demand per Product**")
            time_series_df = df_filtered.groupby(["Month", "Product"])["Product_Demand"].sum().reset_index()
            fig_ts = px.line(time_series_df, x="Month", y="Product_Demand", color="Product", labels={"Product_Demand": "Total Demand"}, template=plotly_template)
            fig_ts.update_layout(legend_title_text='Product')
            st.plotly_chart(fig_ts, use_container_width=True)
            
        with tab2:
            st.markdown("**Demand Distribution by Product, Region, and Segment**")
            cat_col1, cat_col2, cat_col3 = st.columns(3)
            with cat_col1: fig_prod = px.box(df_filtered, x="Product", y="Product_Demand", title="By Product", points="outliers", labels={"Product_Demand": "Demand"}, template=plotly_template); st.plotly_chart(fig_prod, use_container_width=True)
            with cat_col2: fig_reg = px.box(df_filtered, x="Region", y="Product_Demand", title="By Region", points="outliers", labels={"Product_Demand": "Demand"}, template=plotly_template); st.plotly_chart(fig_reg, use_container_width=True)
            with cat_col3: fig_seg = px.box(df_filtered, x="Customer_Segment", y="Product_Demand", title="By Segment", points="outliers", labels={"Product_Demand": "Demand"}, template=plotly_template); st.plotly_chart(fig_seg, use_container_width=True)
            
        with tab3:
            st.markdown("**Correlation Matrix of Key Numerical Features**")
            numerical_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
            corr_cols = ["Product_Demand", "Ad_Spend", "Interest_Rate", "Market_Index", "Previous_Month_Demand"]
            valid_corr_cols = [col for col in corr_cols if col in numerical_cols]
            if len(valid_corr_cols) > 1:
                correlation_matrix = df_filtered[valid_corr_cols].corr()
                fig_corr = go.Figure(data=go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.columns, colorscale='Viridis', zmin=-1, zmax=1, colorbar=dict(title='Correlation')))
                fig_corr.update_layout(template=plotly_template)
                st.plotly_chart(fig_corr, use_container_width=True)
            else: st.warning("Not enough numerical columns for correlation analysis.")
            
    elif df is None: st.warning("Data not loaded. Cannot perform EDA.")
    else: st.warning("No data available for the selected global filters.")

# --- Forecasting & Comparison Page ---
elif st.session_state.page == "📈 Forecasting & Comparison":
    st.title("📈 Forecasting Models & Causal Impact Simulation")
    st.markdown("Evaluate different forecasting approaches and simulate the impact of interventions.")

    if df_filtered is not None and not df_filtered.empty:
        
        forecast_tabs = st.tabs(["Baseline Models", "🧠 Causal Impact Simulator", "✨ Combined Causal ML", "🏆 Model Comparison"])
        
        # --- Baseline Models Tab ---
        with forecast_tabs[0]:
            st.header("Baseline Model Forecasts")
            st.markdown("Performance of traditional forecasting models (Linear Regression, Simple Time Series, XGBoost placeholder) on the test set.")
            
            # Updated model list
            baseline_model_type = st.selectbox("Select Baseline Model", ["Linear Regression", "XGBoost (Placeholder)", "Simple Time Series"], key="baseline_select")
            results_df_baseline = pd.DataFrame()
            rmse_baseline, mape_baseline, model_obj_baseline = np.nan, np.nan, None
            
            if baseline_model_type == "Linear Regression":
                with st.spinner("Training Linear Regression Model..."): model_obj_baseline, rmse_baseline, mape_baseline, results_df_baseline = train_evaluate_linear_regression(df_filtered)
            elif baseline_model_type == "XGBoost (Placeholder)":
                with st.spinner("Training XGBoost (RF Placeholder) Model..."): model_obj_baseline, rmse_baseline, mape_baseline, results_df_baseline = train_evaluate_xgboost(df_filtered)
            elif baseline_model_type == "Simple Time Series":
                # Removed product selection for ARIMA
                with st.spinner(f"Training Simple Time Series Model..."): model_obj_baseline, rmse_baseline, mape_baseline, results_df_baseline = train_evaluate_simple_ts(df_filtered)
            
            st.subheader(f"{baseline_model_type} Performance (Test Set)")
            bl_col1, bl_col2 = st.columns(2)
            with bl_col1: st.metric("RMSE", f"{rmse_baseline:.2f}")
            with bl_col2: st.metric("MAPE", f"{mape_baseline:.2%}")
            
            if not results_df_baseline.empty:
                fig_pred_baseline = go.Figure()
                fig_pred_baseline.add_trace(go.Scatter(x=results_df_baseline["Month"], y=results_df_baseline["Actual"], mode='lines', name='Actual Demand'))
                fig_pred_baseline.add_trace(go.Scatter(x=results_df_baseline["Month"], y=results_df_baseline["Predicted"], mode='lines', name='Predicted Demand', line=dict(dash='dash')))
                fig_pred_baseline.update_layout(title=f"Actual vs. Predicted Demand - {baseline_model_type}", xaxis_title="Month", yaxis_title="Product Demand", template=plotly_template)
                st.plotly_chart(fig_pred_baseline, use_container_width=True)
            else: st.warning("Could not generate predictions for the selected baseline model.")

        # --- Causal Impact Simulator Tab ---
        with forecast_tabs[1]:
            st.header("🧠 Causal Impact Simulator")
            st.markdown("Estimate the causal effect of interventions (like changing Ad Spend or Interest Rates) on Product Demand and run 'What-If' scenarios.")
            
            if not ECONML_AVAILABLE:
                st.warning("⚠️ Causal inference features are disabled in this deployment due to dependency constraints. This is a simplified version of the dashboard.")
                st.info("The full version with causal inference requires the econml package. Consider deploying locally or on a platform that supports all dependencies.")
            else:
                treatment_options = {"Advertising Spend": "Ad_Spend", "Interest Rate": "Interest_Rate", "Promotion Active": "Promotion_Active"}
                selected_treatment_label = st.selectbox("Select Treatment Variable to Analyze", list(treatment_options.keys()), key="causal_treat_select")
                selected_treatment_var = treatment_options[selected_treatment_label]
                outcome_var = "Product_Demand"
                
                st.subheader(f"Estimated Average Effect of {selected_treatment_label}")
                with st.spinner(f"Estimating causal effect for {selected_treatment_label}..."): causal_model, ate, lower_ci, upper_ci, X_test_causal, _ = train_causal_model(df_filtered, selected_treatment_var, outcome_var)
                
                if not np.isnan(ate):
                    st.metric(f"Average Treatment Effect (ATE)", f"{ate:.3f}")
                    st.caption(f"*Interpretation:* A one-unit change in {selected_treatment_label} is estimated to change {outcome_var} by {ate:.3f} units, on average. (95% CI: [{lower_ci:.3f}, {upper_ci:.3f}])")
                else:
                    st.warning(f"Could not estimate ATE for {selected_treatment_label}.")

                st.divider()
                st.subheader("What-If Scenario Simulation")
                if causal_model is not None and X_test_causal is not None:
                    try:
                        categorical_features_cf = ["Product", "Region", "Customer_Segment"]
                        df_encoded_cf = pd.get_dummies(df_filtered, columns=categorical_features_cf, drop_first=True)
                        confounders_cf = get_confounders(df_encoded_cf, selected_treatment_var, outcome_var)
                        X_cf = df_encoded_cf[confounders_cf]
                        T_cf = df_encoded_cf[selected_treatment_var]
                        Y_cf = df_encoded_cf[outcome_var]
                        scaler_cf = StandardScaler()
                        X_scaled_cf = scaler_cf.fit_transform(X_cf)
                        X_scaled_df_cf = pd.DataFrame(X_scaled_cf, columns=confounders_cf, index=X_cf.index)
                        _, X_test_causal_sim, _, T_test_cf, _, _ = train_test_split(X_scaled_df_cf, T_cf, Y_cf, test_size=0.3, random_state=42)
                        baseline_value = T_test_cf.mean()
                    except Exception as e:
                        st.warning(f"Could not determine baseline value for scenario: {e}")
                        baseline_value = df_filtered[selected_treatment_var].mean()
                    
                    st.markdown(f"Simulate the impact of changing **{selected_treatment_label}** from its average test set value ({baseline_value:.2f}).")
                    
                    sim_col1, sim_col2 = st.columns([1,2])
                    with sim_col1:
                        if selected_treatment_var == "Promotion_Active":
                            counterfactual_value = st.radio("Set Promotion Status:", [0, 1], index=int(round(baseline_value)), format_func=lambda x: "OFF" if x == 0 else "ON", key="promo_radio")
                        else:
                            min_val = float(df_filtered[selected_treatment_var].min())
                            max_val = float(df_filtered[selected_treatment_var].max())
                            default_val = float(baseline_value)
                            counterfactual_value = st.number_input(f"Set Hypothetical {selected_treatment_label} Value:", min_value=min_val, max_value=max_val, value=default_val, step=(max_val-min_val)/100, key="counterfactual_input")
                    
                    with sim_col2:
                        with st.container(): # Container for result
                            try:
                                # Ensure X_test_causal_sim is not empty and has the correct shape/index
                                if not X_test_causal_sim.empty:
                                    counterfactual_effect = causal_model.effect(X_test_causal_sim, T0=baseline_value, T1=counterfactual_value)
                                    avg_counterfactual_effect = counterfactual_effect.mean()
                                    st.success(f"**Estimated Impact:**")
                                    st.markdown(f"Changing {selected_treatment_label} from `{baseline_value:.2f}` to `{counterfactual_value:.2f}` is predicted to change average demand by **`{avg_counterfactual_effect:.3f}`** units.")
                                else:
                                    st.warning("Test data for causal model is empty, cannot run simulation.")
                            except Exception as e: 
                                st.error(f"Could not calculate counterfactual effect: {e}")
                                st.caption("This might happen if the model or data is unsuitable for the chosen scenario.")
                else: st.warning("Causal model not available for scenario simulation.")

        # --- Combined Causal ML Tab ---
        with forecast_tabs[2]:
            st.header("✨ Combined Causal + ML Forecast")
            
            if not ECONML_AVAILABLE:
                st.markdown("This model would normally incorporate estimated causal effects as features into a Random Forest model to potentially improve forecast accuracy.")
                st.warning("⚠️ Causal ML features are using standard RandomForest without causal features due to dependency constraints.")
            else:
                st.markdown("This model incorporates estimated causal effects as features into a Random Forest model to potentially improve forecast accuracy.")
            
            results_df_cml = pd.DataFrame()
            rmse_cml, mape_cml, model_obj_cml = np.nan, np.nan, None
            with st.spinner("Training Combined Causal ML Model..."): model_obj_cml, rmse_cml, mape_cml, results_df_cml = train_evaluate_causal_ml(df_filtered)
                
            st.subheader("Combined Model Performance (Test Set)")
            cml_col1, cml_col2 = st.columns(2)
            with cml_col1: st.metric("RMSE", f"{rmse_cml:.2f}")
            with cml_col2: st.metric("MAPE", f"{mape_cml:.2%}")
            
            if not results_df_cml.empty:
                fig_pred_cml = go.Figure()
                fig_pred_cml.add_trace(go.Scatter(x=results_df_cml["Month"], y=results_df_cml["Actual"], mode='lines', name='Actual Demand'))
                fig_pred_cml.add_trace(go.Scatter(x=results_df_cml["Month"], y=results_df_cml["Predicted"], mode='lines', name='Predicted Demand (Causal ML)' , line=dict(dash='dash')))
                fig_pred_cml.update_layout(title="Actual vs. Predicted Demand - Combined Causal ML Model", xaxis_title="Month", yaxis_title="Product Demand", template=plotly_template)
                st.plotly_chart(fig_pred_cml, use_container_width=True)
            else: st.warning("Could not generate predictions for the combined model.")

        # --- Model Comparison Tab ---
        with forecast_tabs[3]:
            st.header("🏆 Model Comparison")
            st.markdown("Comparing the performance of different forecasting models based on the selected filters.")
            
            model_performance = {}
            
            # --- Calculate performance for each model ---
            with st.spinner("Calculating performance for all models..."):
                # Linear Regression (use cached results if available)
                try:
                    if "rmse_baseline" in locals() and baseline_model_type == "Linear Regression":
                        model_performance["Linear Regression"] = {"RMSE": rmse_baseline, "MAPE": mape_baseline}
                    else:
                        _, lr_rmse, lr_mape, _ = train_evaluate_linear_regression(df_filtered)
                        model_performance["Linear Regression"] = {"RMSE": lr_rmse, "MAPE": lr_mape}
                except Exception as e:
                    st.warning(f"Linear Regression failed: {e}")
                    model_performance["Linear Regression"] = {"RMSE": np.nan, "MAPE": np.nan}
                
                # XGBoost (Placeholder) (use cached results if available)
                try:
                    if "rmse_baseline" in locals() and baseline_model_type == "XGBoost (Placeholder)":
                         model_performance["XGBoost (RF)"] = {"RMSE": rmse_baseline, "MAPE": mape_baseline}
                    else:
                        _, xgb_rmse, xgb_mape, _ = train_evaluate_xgboost(df_filtered)
                        model_performance["XGBoost (RF)"] = {"RMSE": xgb_rmse, "MAPE": xgb_mape}
                except Exception as e:
                    st.warning(f"XGBoost (RF) failed: {e}")
                    model_performance["XGBoost (RF)"] = {"RMSE": np.nan, "MAPE": np.nan}

                # Simple Time Series (use cached results if available)
                try:
                    if "rmse_baseline" in locals() and baseline_model_type == "Simple Time Series":
                        model_performance["Simple TS"] = {"RMSE": rmse_baseline, "MAPE": mape_baseline}
                    else:
                        _, ts_rmse, ts_mape, _ = train_evaluate_simple_ts(df_filtered)
                        model_performance["Simple TS"] = {"RMSE": ts_rmse, "MAPE": ts_mape}
                except Exception as e:
                    st.warning(f"Simple Time Series failed: {e}")
                    model_performance["Simple TS"] = {"RMSE": np.nan, "MAPE": np.nan}
                    
                # Combined Causal ML (use cached results if available)
                try:
                    if "rmse_cml" in locals():
                         model_performance["Causal ML"] = {"RMSE": rmse_cml, "MAPE": mape_cml}
                    else:
                        _, cml_rmse_comp, cml_mape_comp, _ = train_evaluate_causal_ml(df_filtered)
                        model_performance["Causal ML"] = {"RMSE": cml_rmse_comp, "MAPE": cml_mape_comp}
                except Exception as e:
                    st.warning(f"Causal ML failed: {e}")
                    model_performance["Causal ML"] = {"RMSE": np.nan, "MAPE": np.nan}
            
            # --- Display Results ---
            st.subheader("Performance Metrics Summary")
            
            perf_df = pd.DataFrame(model_performance).T # Transpose for better viewing
            # Ensure columns exist before formatting
            if "MAPE" in perf_df.columns: perf_df["MAPE"] = perf_df["MAPE"].map('{:.2%}'.format)
            if "RMSE" in perf_df.columns: perf_df["RMSE"] = perf_df["RMSE"].map('{:.2f}'.format)
            st.dataframe(perf_df, use_container_width=True)
            
            st.divider()
            st.subheader("Visual Comparison")
            
            # Prepare data for plotting (handle NaNs)
            plot_df = pd.DataFrame(model_performance).T.reset_index()
            plot_df.rename(columns={"index": "Model"}, inplace=True)
            plot_df_melt = pd.melt(plot_df, id_vars="Model", var_name="Metric", value_name="Value")
            plot_df_melt = plot_df_melt.dropna(subset=["Value"])
            
            if not plot_df_melt.empty:
                fig_comp = px.bar(plot_df_melt, x="Model", y="Value", color="Model", 
                                  facet_row="Metric", title="Model Performance Comparison",
                                  labels={"Value": "Metric Value"}, template=plotly_template)
                fig_comp.update_yaxes(matches=None) # Allow different y-axis scales for RMSE and MAPE
                fig_comp.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("Could not generate comparison chart due to missing model results.")

    elif df is None: st.warning("Data not loaded. Cannot compare models.")
    else: st.warning("No data available for the selected global filters.")

# --- Download & Links Page ---
elif st.session_state.page == "📥 Download & Links":
    st.title("📥 Download Data & Project Links")
    st.markdown("Access the dataset used in this application and find links to the project resources.")
    
    with st.container():
        st.subheader("Download Dataset")
        if df is not None:
            st.markdown(get_table_download_link(df, filename="simulated_banking_demand_dataset.csv", text="Download Full Simulated Dataset (CSV)"), unsafe_allow_html=True)
        else:
            st.warning("Original dataset not loaded.")
            
    st.divider()
    with st.container():
        st.subheader("Project Resources")
        st.markdown("""
        *   **GitHub Repository:** [Link to your GitHub repo] (Replace with actual link)
        *   **Project Report/Paper:** [Link to your report/paper] (Replace with actual link)
        *   **My Portfolio/Resume:** [Link to your portfolio/resume] (Replace with actual link)
        """, unsafe_allow_html=True)
        st.info("Note: Please replace the bracketed links above with your actual URLs.")

# --- Fallback for invalid state ---
else:
    st.error("Invalid page selected. Please use the sidebar navigation.")
    st.button("Go to Home", on_click=set_page, args=("🏠 Home / Overview",))

