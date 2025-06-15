import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌍 Global Sustainable Energy Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🌍 Global Sustainable Energy Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Explore sustainable energy trends, predict future patterns, and discover insights from global data (2000-2020)")

# File upload function (without caching)
@st.cache_data
def process_uploaded_data(uploaded_file):
    """Process and cache the uploaded dataset"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# File upload interface (not cached)
def display_upload_interface():
    """Display file upload interface and sample data download"""
    uploaded_file = st.file_uploader("Upload your Sustainable Energy Dataset (CSV)", type=['csv'])
    
    # Add sample dataset download button
    st.markdown("### 📥 Download Dataset:")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        # Direct repository dataset download button
        repo_data_url = "https://github.com/Shreyas521032/EcoVision-Sustainable-Energy-Analytics-Platform/blob/main/Dataset/global-data-on-sustainable-energy.csv"
        st.markdown("""
        <a href="https://github.com/Shreyas521032/EcoVision-Sustainable-Energy-Analytics-Platform/blob/main/Dataset/global-data-on-sustainable-energy.csv" download="global-data-on-sustainable-energy.csv" target="_blank">
            <button style="
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                font-size: 14px;
                transition: background-color 0.3s;
            " onmouseover="this.style.backgroundColor='#45a049'" onmouseout="this.style.backgroundColor='#4CAF50'">
                📥 Download Dataset
            </button>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        **Dataset:** Global Data on Sustainable Energy (2000-2020)  
        **Source:** Repository Dataset  
        **Size:** ~200 countries, 21 years of data  
        **Format:** CSV file ready for analysis
        
        **📋 Instructions:**
        1. Click "📥 Download Dataset CSV" to download the file
        2. Then upload it using the file uploader above
        
        **OR**
        
        1. Click "🔄 Load Sample Data" to load directly from repository
        """)
    
    st.markdown("---")
    
    return uploaded_file

# Display upload interface and load data
uploaded_file = display_upload_interface()

# Process data if file is uploaded or loaded from repository
if uploaded_file is not None:
    if isinstance(uploaded_file, str):
        # If it's a file path (loaded from repository)
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset loaded successfully from repository!")
        except Exception as e:
            st.error(f"Error loading data from repository: {str(e)}")
            df = None
    else:
        # If it's an uploaded file
        df = process_uploaded_data(uploaded_file)
    
    if df is None:
        st.stop()
else:
    df = None
    st.warning("Please upload your sustainable energy dataset CSV file or load the sample data.")
    st.info("Expected columns: Entity, Year, Access to electricity (% of population), etc.")

if df is not None:
    # Data preprocessing
    @st.cache_data
    def preprocess_data(df):
        """Clean and preprocess the dataset"""
        # Handle missing values and clean column names
        df_clean = df.copy()
        
        # Standardize column names (remove special characters and spaces)
        column_mapping = {
            'Access to electricity (% of population)': 'electricity_access',
            'Access to clean fuels for cooking (% of population)': 'clean_cooking_access',
            'Renewable-electricity-generating-capacity-per-capita': 'renewable_capacity_per_capita',
            'Financial flows to developing countries (US $)': 'financial_flows',
            'Renewable energy share in total final energy consumption (%)': 'renewable_share',
            'Electricity from fossil fuels (TWh)': 'fossil_electricity',
            'Electricity from nuclear (TWh)': 'nuclear_electricity',
            'Electricity from renewables (TWh)': 'renewable_electricity',
            'Low-carbon electricity (% electricity)': 'low_carbon_percentage',
            'Primary energy consumption per capita (kWh/person)': 'energy_per_capita',
            'Energy intensity level of primary energy (MJ/$2011 PPP GDP)': 'energy_intensity',
            'Value_co2_emissions (metric tons per capita)': 'co2_per_capita',
            'Renewables (% equivalent primary energy)': 'renewables_primary_energy',
            'GDP growth (annual %)': 'gdp_growth',
            'GDP per capita': 'gdp_per_capita',
            'Density (P/Km2)': 'population_density',
            'Land Area (Km2)': 'land_area',
            'Latitude': 'latitude',
            'Longitude': 'longitude'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df_clean.columns:
                df_clean = df_clean.rename(columns={old_name: new_name})
        
        # Fill missing values with appropriate methods
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Year':
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        return df_clean
    
    df_clean = preprocess_data(df)
    
    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📊 Overview & Key Metrics", "🌍 Geographic Analysis", "📈 Trend Analysis", 
         "🔄 Correlation Analysis", "🤖 Predictive Modeling", "🔮 Time Series Forecasting"]
    )
    
    # Sidebar filters
    st.sidebar.title("🔧 Filters")
    
    # Year range filter
    min_year, max_year = int(df_clean['Year'].min()), int(df_clean['Year'].max())
    year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    
    # Country filter
    countries = ['All'] + sorted(df_clean['Entity'].unique().tolist())
    selected_countries = st.sidebar.multiselect("Select Countries", countries, default=['All'])
    
    # Filter data based on selections
    df_filtered = df_clean[
        (df_clean['Year'] >= year_range[0]) & 
        (df_clean['Year'] <= year_range[1])
    ]
    
    if 'All' not in selected_countries and selected_countries:
        df_filtered = df_filtered[df_filtered['Entity'].isin(selected_countries)]
    
    # PAGE 1: Overview & Key Metrics
    if page == "📊 Overview & Key Metrics":
        st.header("📊 Dataset Overview & Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Countries", len(df_filtered['Entity'].unique()))
        with col2:
            st.metric("Years Covered", f"{df_filtered['Year'].min()}-{df_filtered['Year'].max()}")
        with col3:
            if 'co2_per_capita' in df_filtered.columns:
                avg_co2 = df_filtered['co2_per_capita'].mean()
                st.metric("Avg CO2 per Capita", f"{avg_co2:.2f} tons")
        with col4:
            if 'renewable_share' in df_filtered.columns:
                avg_renewable = df_filtered['renewable_share'].mean()
                st.metric("Avg Renewable Share", f"{avg_renewable:.1f}%")
        
        # Dataset info
        st.subheader("📋 Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", df_filtered.shape)
            st.write("**Countries:**", len(df_filtered['Entity'].unique()))
            st.write("**Time Period:**", f"{df_filtered['Year'].min()} - {df_filtered['Year'].max()}")
        
        with col2:
            st.write("**Key Indicators Available:**")
            indicators = [col for col in df_filtered.columns if col not in ['Entity', 'Year', 'latitude', 'longitude']]
            st.write(f"- {len(indicators)} sustainability indicators")
            st.write("- Energy access metrics")
            st.write("- Renewable energy data")
            st.write("- Economic indicators")
        
        # Top/Bottom performers
        st.subheader("🏆 Top & Bottom Performers (Latest Year)")
        
        try:
            latest_year = df_filtered['Year'].max()
            latest_data = df_filtered[df_filtered['Year'] == latest_year].copy()
            
            if len(latest_data) == 0:
                st.warning("No data available for the latest year in your filtered selection.")
            else:
                st.info(f"📅 Showing data for year: **{latest_year}** ({len(latest_data)} countries)")
                
                # Check for renewable energy data
                renewable_cols = [
                    'renewable_share', 
                    'Renewable energy share in total final energy consumption (%)', 
                    'renewables_primary_energy',
                    'Renewables (% equivalent primary energy)',
                    'renewable_energy_share',
                    'Renewable energy share',
                    'renewables_percentage',
                    'renewable_percentage'
                ]
                renewable_col = None
                
                # Check each possible renewable energy column name
                for col in renewable_cols:
                    if col in latest_data.columns:
                        renewable_col = col
                        break
                
                # If not found, search for any column containing 'renewable'
                if renewable_col is None:
                    for col in latest_data.columns:
                        col_lower = col.lower()
                        if 'renewable' in col_lower and ('share' in col_lower or '%' in col_lower or 'percent' in col_lower):
                            renewable_col = col
                            st.info(f"🔍 Found renewable energy column: '{col}'")
                            break
                
                if renewable_col is not None:
                    st.write("**🌟 Top 10 Countries - Renewable Energy Share**")
                    # Remove NaN values and get top performers
                    renewable_data = latest_data.dropna(subset=[renewable_col])
                    if len(renewable_data) > 0:
                        top_renewable = renewable_data.nlargest(10, renewable_col)[['Entity', renewable_col]]
                        if len(top_renewable) > 0:
                            fig = px.bar(top_renewable, x=renewable_col, y='Entity', orientation='h',
                                       color=renewable_col, color_continuous_scale='Greens',
                                       title="Top 10 Countries by Renewable Energy Share",
                                       labels={renewable_col: 'Renewable Energy Share (%)'})
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No renewable energy data available for visualization.")
                    else:
                        st.warning("No valid renewable energy data found.")
                else:
                    st.warning("⚠️ Renewable energy share column not found in dataset.")
                    
                    # Show columns that might be renewable-related
                    possible_renewable_cols = [col for col in latest_data.columns 
                                             if 'renewable' in col.lower()]
                    
                    if possible_renewable_cols:
                        st.write("**Possible renewable energy columns found:**")
                        for col in possible_renewable_cols:
                            st.write(f"- `{col}`")
                    else:
                        st.write("**No renewable energy columns detected.**")
                        st.write("**All numeric columns:**")
                        numeric_cols = latest_data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if col != 'Year':
                                st.write(f"- `{col}`")
                
                # Show actual column names for debugging
                with st.expander("🔍 Debug: Available Columns in Dataset"):
                    st.write("**All columns in the dataset:**")
                    for i, col in enumerate(latest_data.columns, 1):
                        st.write(f"{i}. `{col}`")
                    
                    st.write(f"**Latest year:** {latest_year}")
                    st.write(f"**Countries in latest year:** {len(latest_data)}")
        
        except Exception as e:
            st.error(f"Error in Top & Bottom Performers section: {str(e)}")
            st.write("**Debug info:**")
            st.write(f"- Dataset shape: {df_filtered.shape}")
            st.write(f"- Available columns: {list(df_filtered.columns)}")
            st.write(f"- Year range: {df_filtered['Year'].min()} to {df_filtered['Year'].max()}")
    
    # PAGE 2: Geographic Analysis
    elif page == "🌍 Geographic Analysis":
        st.header("🌍 Geographic Analysis")
        
        # World map visualizations
        if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
            latest_year = df_filtered['Year'].max()
            map_data = df_filtered[df_filtered['Year'] == latest_year]
            
            # Select metric for mapping
            numeric_cols = [col for col in map_data.select_dtypes(include=[np.number]).columns 
                          if col not in ['Year', 'latitude', 'longitude']]
            
            if numeric_cols:
                selected_metric = st.selectbox("Select Metric for World Map", numeric_cols)
                
                # Create world map
                fig = px.scatter_geo(
                    map_data,
                    lat='latitude',
                    lon='longitude',
                    color=selected_metric,
                    size=selected_metric,
                    hover_name='Entity',
                    hover_data=[selected_metric],
                    title=f"Global Distribution of {selected_metric.replace('_', ' ').title()} ({latest_year})",
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        # Regional analysis
        st.subheader("🌏 Regional Comparisons")
        
        # Create regional groupings (simplified)
        def assign_region(country):
            europe = ['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Netherlands', 'Belgium', 'Sweden', 'Norway', 'Denmark']
            asia = ['China', 'India', 'Japan', 'South Korea', 'Indonesia', 'Thailand', 'Malaysia', 'Singapore']
            north_america = ['United States', 'Canada', 'Mexico']
            south_america = ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru']
            africa = ['South Africa', 'Nigeria', 'Kenya', 'Ghana', 'Morocco', 'Egypt']
            oceania = ['Australia', 'New Zealand']
            
            if country in europe:
                return 'Europe'
            elif country in asia:
                return 'Asia'
            elif country in north_america:
                return 'North America'
            elif country in south_america:
                return 'South America'
            elif country in africa:
                return 'Africa'
            elif country in oceania:
                return 'Oceania'
            else:
                return 'Other'
        
        df_regional = df_filtered.copy()
        df_regional['Region'] = df_regional['Entity'].apply(assign_region)
        
        # Regional comparison charts
        col1, col2 = st.columns(2)
        
        if 'renewable_share' in df_regional.columns:
            with col1:
                regional_renewable = df_regional.groupby(['Region', 'Year'])['renewable_share'].mean().reset_index()
                fig = px.line(regional_renewable, x='Year', y='renewable_share', color='Region',
                            title="Renewable Energy Share by Region Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        if 'co2_per_capita' in df_regional.columns:
            with col2:
                regional_co2 = df_regional.groupby(['Region', 'Year'])['co2_per_capita'].mean().reset_index()
                fig = px.line(regional_co2, x='Year', y='co2_per_capita', color='Region',
                            title="CO2 per Capita by Region Over Time")
                st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 3: Trend Analysis
    elif page == "📈 Trend Analysis":
        st.header("📈 Trend Analysis Over Time")
        
        # Global trends
        st.subheader("🌍 Global Trends")
        
        # Calculate global averages by year
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        trend_cols = [col for col in numeric_cols if col != 'Year']
        
        if trend_cols:
            selected_trends = st.multiselect(
                "Select Indicators for Trend Analysis", 
                trend_cols, 
                default=trend_cols[:4] if len(trend_cols) >= 4 else trend_cols
            )
            
            if selected_trends:
                global_trends = df_filtered.groupby('Year')[selected_trends].mean().reset_index()
                
                # Create subplots
                rows = (len(selected_trends) + 1) // 2
                fig = make_subplots(
                    rows=rows, cols=2,
                    subplot_titles=selected_trends,
                    vertical_spacing=0.1
                )
                
                for i, col in enumerate(selected_trends):
                    row = (i // 2) + 1
                    col_pos = (i % 2) + 1
                    
                    fig.add_trace(
                        go.Scatter(x=global_trends['Year'], y=global_trends[col], 
                                 mode='lines+markers', name=col.replace('_', ' ').title()),
                        row=row, col=col_pos
                    )
                
                fig.update_layout(height=300*rows, showlegend=False, title_text="Global Trends Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        # Country-specific trends
        st.subheader("🏳️ Country-Specific Trends")
        
        if len(df_filtered['Entity'].unique()) > 1:
            selected_countries_trend = st.multiselect(
                "Select Countries for Comparison", 
                df_filtered['Entity'].unique(),
                default=list(df_filtered['Entity'].unique())[:5]
            )
            
            selected_indicator = st.selectbox(
                "Select Indicator for Country Comparison",
                trend_cols
            )
            
            if selected_countries_trend and selected_indicator:
                country_trends = df_filtered[df_filtered['Entity'].isin(selected_countries_trend)]
                
                fig = px.line(country_trends, x='Year', y=selected_indicator, color='Entity',
                            title=f"{selected_indicator.replace('_', ' ').title()} by Country Over Time",
                            markers=True)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # Growth rate analysis
        st.subheader("📊 Growth Rate Analysis")
        
        growth_indicator = st.selectbox("Select Indicator for Growth Analysis", trend_cols)
        
        if growth_indicator:
            # Calculate year-over-year growth rates
            growth_data = []
            for country in df_filtered['Entity'].unique():
                country_data = df_filtered[df_filtered['Entity'] == country].sort_values('Year')
                if len(country_data) > 1:
                    country_data[f'{growth_indicator}_growth'] = country_data[growth_indicator].pct_change() * 100
                    growth_data.append(country_data)
            
            if growth_data:
                growth_df = pd.concat(growth_data)
                avg_growth = growth_df.groupby('Year')[f'{growth_indicator}_growth'].mean().reset_index()
                
                fig = px.line(avg_growth, x='Year', y=f'{growth_indicator}_growth',
                            title=f"Average Annual Growth Rate - {growth_indicator.replace('_', ' ').title()}")
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 4: Correlation Analysis
    elif page == "🔄 Correlation Analysis":
        st.header("🔄 Correlation Analysis")
        
        # Correlation matrix
        st.subheader("🎯 Correlation Matrix")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        correlation_cols = [col for col in numeric_cols if col != 'Year']
        
        if len(correlation_cols) > 1:
            selected_corr_cols = st.multiselect(
                "Select Variables for Correlation Analysis",
                correlation_cols,
                default=correlation_cols[:8] if len(correlation_cols) >= 8 else correlation_cols
            )
            
            if len(selected_corr_cols) > 1:
                corr_matrix = df_filtered[selected_corr_cols].corr()
                
                # Create heatmap
                fig = px.imshow(corr_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              color_continuous_scale='RdBu_r',
                              title="Correlation Matrix of Sustainability Indicators")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Key correlations
                st.subheader("🔍 Key Correlations")
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        var1, var2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        corr_pairs.append((var1, var2, corr_val))
                
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🔗 Strongest Positive Correlations:**")
                    positive_corrs = [cp for cp in corr_pairs if cp[2] > 0][:5]
                    for var1, var2, corr in positive_corrs:
                        st.write(f"• {var1.replace('_', ' ').title()} ↔ {var2.replace('_', ' ').title()}: **{corr:.3f}**")
                
                with col2:
                    st.write("**🔀 Strongest Negative Correlations:**")
                    negative_corrs = [cp for cp in corr_pairs if cp[2] < 0][:5]
                    for var1, var2, corr in negative_corrs:
                        st.write(f"• {var1.replace('_', ' ').title()} ↔ {var2.replace('_', ' ').title()}: **{corr:.3f}**")
        
        # Scatter plot analysis
        st.subheader("📊 Scatter Plot Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Select X Variable", correlation_cols, key='x_var')
        with col2:
            y_var = st.selectbox("Select Y Variable", correlation_cols, key='y_var')
        
        if x_var != y_var:
            latest_year = df_filtered['Year'].max()
            scatter_data = df_filtered[df_filtered['Year'] == latest_year]
            
            fig = px.scatter(scatter_data, x=x_var, y=y_var, hover_name='Entity',
                           title=f"{y_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()} ({latest_year})",
                           trendline="ols")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 5: Predictive Modeling
    elif page == "🤖 Predictive Modeling":
        st.header("🤖 Predictive Modeling")
        
        st.subheader("🎯 Model Configuration")
        
        # Select target variable
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        target_cols = [col for col in numeric_cols if col != 'Year']
        
        target_var = st.selectbox("Select Target Variable to Predict", target_cols)
        
        # Select feature variables
        feature_cols = [col for col in target_cols if col != target_var]
        selected_features = st.multiselect(
            "Select Feature Variables",
            feature_cols,
            default=feature_cols[:5] if len(feature_cols) >= 5 else feature_cols
        )
        
        if target_var and selected_features:
            # Prepare data for modeling
            model_data = df_filtered[['Entity', 'Year', target_var] + selected_features].dropna()
            
            if len(model_data) > 10:  # Minimum data requirement
                st.subheader("📈 Model Training & Evaluation")
                
                # Split data (use earlier years for training, later for testing)
                split_year = int(model_data['Year'].quantile(0.8))
                train_data = model_data[model_data['Year'] <= split_year]
                test_data = model_data[model_data['Year'] > split_year]
                
                X_train = train_data[selected_features]
                y_train = train_data[target_var]
                X_test = test_data[selected_features]
                y_test = test_data[target_var]
                
                if len(X_train) > 0 and len(X_test) > 0:
                    # Train models
                    models = {
                        'Linear Regression': LinearRegression(),
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
                    }
                    
                    model_results = {}
                    
                    for name, model in models.items():
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        model_results[name] = {
                            'model': model,
                            'predictions': y_pred,
                            'MAE': mae,
                            'MSE': mse,
                            'R²': r2
                        }
                    
                    # Display model performance
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🎯 Model Performance Comparison:**")
                        performance_df = pd.DataFrame({
                            'Model': list(model_results.keys()),
                            'MAE': [results['MAE'] for results in model_results.values()],
                            'MSE': [results['MSE'] for results in model_results.values()],
                            'R²': [results['R²'] for results in model_results.values()]
                        })
                        st.dataframe(performance_df)
                    
                    with col2:
                        # Feature importance for Random Forest
                        if 'Random Forest' in model_results:
                            rf_model = model_results['Random Forest']['model']
                            feature_importance = pd.DataFrame({
                                'Feature': selected_features,
                                'Importance': rf_model.feature_importances_
                            }).sort_values('Importance', ascending=True)
                            
                            fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                                       title="Feature Importance (Random Forest)")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction vs Actual plots
                    st.subheader("📊 Prediction vs Actual Values")
                    
                    for name, results in model_results.items():
                        fig = go.Figure()
                        
                        # Add scatter plot
                        fig.add_trace(go.Scatter(
                            x=y_test, 
                            y=results['predictions'],
                            mode='markers',
                            name='Predictions',
                            text=[f"Country: {country}" for country in test_data['Entity']],
                            hovertemplate='Actual: %{x}<br>Predicted: %{y}<br>%{text}<extra></extra>'
                        ))
                        
                        # Add perfect prediction line
                        min_val, max_val = min(y_test.min(), results['predictions'].min()), max(y_test.max(), results['predictions'].max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        ))
                        
                        fig.update_layout(
                            title=f"{name} - Predicted vs Actual {target_var.replace('_', ' ').title()}",
                            xaxis_title=f"Actual {target_var.replace('_', ' ').title()}",
                            yaxis_title=f"Predicted {target_var.replace('_', ' ').title()}",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("Insufficient data for train-test split. Please select a different time range or variables.")
            else:
                st.warning("Insufficient data for modeling. Please select different variables or expand the dataset.")
    
    # PAGE 6: Time Series Forecasting
    elif page == "🔮 Time Series Forecasting":
        st.header("🔮 Time Series Forecasting")
        
        st.subheader("⚙️ Forecasting Configuration")
        
        # Select countries and indicators for forecasting
        forecast_countries = st.multiselect(
            "Select Countries for Forecasting",
            df_filtered['Entity'].unique(),
            default=list(df_filtered['Entity'].unique())[:3]
        )
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        forecast_indicators = [col for col in numeric_cols if col != 'Year']
        
        selected_indicator = st.selectbox("Select Indicator to Forecast", forecast_indicators)
        
        # Forecasting parameters
        col1, col2 = st.columns(2)
        with col1:
            forecast_years = st.slider("Years to Forecast", 1, 10, 5)
        with col2:
            model_type = st.selectbox("Select Forecasting Model", ["Linear Trend", "Polynomial Trend"])
        
        if forecast_countries and selected_indicator:
            st.subheader("📈 Forecasting Results")
            
            for country in forecast_countries:
                country_data = df_filtered[df_filtered['Entity'] == country].sort_values('Year')
                
                if len(country_data) >= 3:  # Minimum data points for forecasting
                    # Prepare time series data
                    years = country_data['Year'].values
                    values = country_data[selected_indicator].values
                    
                    # Remove NaN values
                    valid_idx = ~np.isnan(values)
                    years_clean = years[valid_idx]
                    values_clean = values[valid_idx]
                    
                    if len(years_clean) >= 3:
                        # Fit model based on selection
                        if model_type == "Linear Trend":
                            # Linear regression
                            X = years_clean.reshape(-1, 1)
                            model = LinearRegression()
                            model.fit(X, values_clean)
                            
                            # Generate future years
                            future_years = np.arange(years_clean.max() + 1, years_clean.max() + 1 + forecast_years)
                            future_X = future_years.reshape(-1, 1)
                            future_predictions = model.predict(future_X)
                            
                            # Get historical predictions for plotting
                            historical_predictions = model.predict(X)
                            
                        else:  # Polynomial Trend
                            # Polynomial regression (degree 2)
                            X = years_clean.reshape(-1, 1)
                            poly_features = np.column_stack([X, X**2])
                            model = LinearRegression()
                            model.fit(poly_features, values_clean)
                            
                            # Generate future predictions
                            future_years = np.arange(years_clean.max() + 1, years_clean.max() + 1 + forecast_years)
                            future_X = future_years.reshape(-1, 1)
                            future_poly_features = np.column_stack([future_X, future_X**2])
                            future_predictions = model.predict(future_poly_features)
                            
                            # Get historical predictions for plotting
                            historical_predictions = model.predict(poly_features)
                        
                        # Create forecast plot
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=years_clean,
                            y=values_clean,
                            mode='lines+markers',
                            name='Historical Data',
                            line=dict(color='blue')
                        ))
                        
                        # Historical trend line
                        fig.add_trace(go.Scatter(
                            x=years_clean,
                            y=historical_predictions,
                            mode='lines',
                            name='Historical Trend',
                            line=dict(color='orange', dash='dash')
                        ))
                        
                        # Future predictions
                        fig.add_trace(go.Scatter(
                            x=future_years,
                            y=future_predictions,
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='red', dash='dot'),
                            marker=dict(symbol='diamond')
                        ))
                        
                        # Add vertical line to separate historical and forecast
                        fig.add_vline(
                            x=years_clean.max(),
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Forecast Start"
                        )
                        
                        fig.update_layout(
                            title=f"{country} - {selected_indicator.replace('_', ' ').title()} Forecast",
                            xaxis_title="Year",
                            yaxis_title=selected_indicator.replace('_', ' ').title(),
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast values
                        forecast_df = pd.DataFrame({
                            'Year': future_years,
                            'Predicted Value': future_predictions
                        })
                        
                        with st.expander(f"📊 {country} - Detailed Forecast Values"):
                            st.dataframe(forecast_df)
                            
                            # Calculate trend analysis
                            if len(values_clean) > 1:
                                recent_trend = (values_clean[-1] - values_clean[0]) / len(values_clean)
                                forecast_trend = (future_predictions[-1] - values_clean[-1]) / forecast_years
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Historical Trend (per year)", f"{recent_trend:.3f}")
                                with col2:
                                    st.metric("Forecasted Trend (per year)", f"{forecast_trend:.3f}")
                    
                    else:
                        st.warning(f"Insufficient valid data for {country}")
                else:
                    st.warning(f"Insufficient data for {country} (need at least 3 data points)")
            
            # Global forecast summary
            st.subheader("🌍 Global Forecast Summary")
            
            if len(forecast_countries) > 1:
                # Create comparison chart
                comparison_data = []
                
                for country in forecast_countries:
                    country_data = df_filtered[df_filtered['Entity'] == country].sort_values('Year')
                    
                    if len(country_data) >= 3:
                        years = country_data['Year'].values
                        values = country_data[selected_indicator].values
                        
                        valid_idx = ~np.isnan(values)
                        years_clean = years[valid_idx]
                        values_clean = values[valid_idx]
                        
                        if len(years_clean) >= 3:
                            # Simple linear forecast for comparison
                            X = years_clean.reshape(-1, 1)
                            model = LinearRegression()
                            model.fit(X, values_clean)
                            
                            # Predict for next 5 years
                            future_years = np.arange(years_clean.max() + 1, years_clean.max() + 6)
                            future_X = future_years.reshape(-1, 1)
                            future_predictions = model.predict(future_X)
                            
                            # Store results
                            for year, pred in zip(future_years, future_predictions):
                                comparison_data.append({
                                    'Country': country,
                                    'Year': year,
                                    'Predicted_Value': pred
                                })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    fig = px.line(comparison_df, x='Year', y='Predicted_Value', color='Country',
                                title=f"Comparative Forecast - {selected_indicator.replace('_', ' ').title()}",
                                markers=True)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Scenario analysis
        st.subheader("🎭 Scenario Analysis")
        
        st.info("💡 **What-If Analysis**: Explore how different growth rates might affect future outcomes")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            optimistic_growth = st.number_input("Optimistic Annual Growth (%)", -10.0, 20.0, 2.0, 0.1)
        with col2:
            realistic_growth = st.number_input("Realistic Annual Growth (%)", -10.0, 20.0, 1.0, 0.1)
        with col3:
            pessimistic_growth = st.number_input("Pessimistic Annual Growth (%)", -10.0, 20.0, 0.0, 0.1)
        
        if st.button("🔄 Generate Scenarios"):
            if forecast_countries and selected_indicator:
                scenario_country = forecast_countries[0]  # Use first selected country
                country_data = df_filtered[df_filtered['Entity'] == scenario_country].sort_values('Year')
                
                if len(country_data) > 0:
                    latest_value = country_data[selected_indicator].iloc[-1]
                    latest_year = country_data['Year'].iloc[-1]
                    
                    if not np.isnan(latest_value):
                        # Generate scenarios
                        future_years = np.arange(latest_year + 1, latest_year + forecast_years + 1)
                        
                        scenarios = {
                            'Optimistic': [],
                            'Realistic': [],
                            'Pessimistic': []
                        }
                        
                        growth_rates = {
                            'Optimistic': optimistic_growth / 100,
                            'Realistic': realistic_growth / 100,
                            'Pessimistic': pessimistic_growth / 100
                        }
                        
                        for scenario, growth_rate in growth_rates.items():
                            current_value = latest_value
                            scenario_values = [current_value]
                            
                            for year in future_years:
                                current_value *= (1 + growth_rate)
                                scenario_values.append(current_value)
                            
                            scenarios[scenario] = scenario_values[1:]  # Remove initial value
                        
                        # Create scenario plot
                        fig = go.Figure()
                        
                        colors = {'Optimistic': 'green', 'Realistic': 'blue', 'Pessimistic': 'red'}
                        
                        for scenario, values in scenarios.items():
                            fig.add_trace(go.Scatter(
                                x=future_years,
                                y=values,
                                mode='lines+markers',
                                name=f"{scenario} ({growth_rates[scenario]*100:.1f}% growth)",
                                line=dict(color=colors[scenario])
                            ))
                        
                        # Add current value as starting point
                        fig.add_trace(go.Scatter(
                            x=[latest_year],
                            y=[latest_value],
                            mode='markers',
                            name='Current Value',
                            marker=dict(size=12, color='black', symbol='star')
                        ))
                        
                        fig.update_layout(
                            title=f"Scenario Analysis - {scenario_country} - {selected_indicator.replace('_', ' ').title()}",
                            xaxis_title="Year",
                            yaxis_title=selected_indicator.replace('_', ' ').title(),
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Scenario summary table
                        scenario_summary = pd.DataFrame({
                            'Scenario': list(scenarios.keys()),
                            f'Value in {future_years[-1]}': [scenarios[s][-1] for s in scenarios.keys()],
                            'Total Change': [scenarios[s][-1] - latest_value for s in scenarios.keys()],
                            'Percentage Change': [((scenarios[s][-1] - latest_value) / latest_value) * 100 for s in scenarios.keys()]
                        })
                        
                        st.subheader("📋 Scenario Summary")
                        st.dataframe(scenario_summary)

    # Additional insights and recommendations
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Insights & Tips")
    st.sidebar.info("""
    **Key Features:**
    - 📊 Interactive visualizations
    - 🌍 Geographic analysis
    - 📈 Trend analysis
    - 🔄 Correlation studies
    - 🤖 ML predictions
    - 🔮 Time series forecasting
    
    **Tips:**
    - Use filters to focus on specific countries/years
    - Compare multiple indicators for deeper insights
    - Check correlations to understand relationships
    - Use forecasting for future planning
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666;'>
        <p>🌱 <strong>Sustainable Energy Analytics Dashboard</strong> 🌱</p>
        <p>Built with Streamlit • Data: Global Sustainable Energy (2000-2020)</p>
        <p>💡 Empowering data-driven decisions for a sustainable future</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Instructions for users when no data is loaded
    st.markdown("""
    ## 📋 Instructions
    
    To use this dashboard:
    
    1. **Upload your dataset**: Click "Browse files" above and upload your sustainable energy CSV file
    2. **Expected format**: The CSV should contain columns like:
       - Entity (country names)
       - Year (2000-2020)
       - Access to electricity (% of population)
       - Renewable energy share in total final energy consumption (%)
       - CO2 emissions per capita
       - And other sustainability indicators
    
    3. **Features available after upload**:
       - 📊 **Overview & Key Metrics**: Dataset summary and top performers
       - 🌍 **Geographic Analysis**: World maps and regional comparisons
       - 📈 **Trend Analysis**: Time series trends and growth rates
       - 🔄 **Correlation Analysis**: Relationship between variables
       - 🤖 **Predictive Modeling**: ML models for prediction
       - 🔮 **Time Series Forecasting**: Future predictions and scenarios
    
    ## 🎯 Sample Data Structure
    
    Your CSV should look like this:
    
    | Entity | Year | Access to electricity (% of population) | Renewable energy share (%) | CO2 emissions per capita |
    |--------|------|----------------------------------------|---------------------------|-------------------------|
    | Germany| 2020 | 100.0                                  | 17.4                      | 8.7                     |
    | India  | 2020 | 95.2                                   | 9.9                       | 1.9                     |
    
    ## 🚀 Getting Started
    
    1. Download the "Global Data on Sustainable Energy (2000-2020)" dataset from Kaggle
    2. Upload it using the file uploader above
    3. Explore the various analysis pages using the sidebar navigation
    4. Use filters to focus on specific countries or time periods
    5. Generate insights and forecasts for your sustainability analysis
    
    ## 📈 Key Capabilities
    
    - **Interactive Visualizations**: Plotly-powered charts and maps
    - **Machine Learning**: Prediction models using scikit-learn
    - **Time Series Forecasting**: Future trend predictions
    - **Scenario Analysis**: What-if simulations
    - **Comprehensive Analytics**: From basic stats to advanced modeling
    """)
    
    # Sample data preview
    st.markdown("""
    ## 📊 Expected Data Columns
    
    The dashboard expects these columns in your dataset:
    """)
    
    sample_columns = [
        "Entity", "Year", "Access to electricity (% of population)",
        "Access to clean fuels for cooking (% of population)",
        "Renewable-electricity-generating-capacity-per-capita",
        "Financial flows to developing countries (US $)",
        "Renewable energy share in total final energy consumption (%)",
        "Electricity from fossil fuels (TWh)", "Electricity from nuclear (TWh)",
        "Electricity from renewables (TWh)", "Low-carbon electricity (% electricity)",
        "Primary energy consumption per capita (kWh/person)",
        "Energy intensity level of primary energy (MJ/$2011 PPP GDP)",
        "Value_co2_emissions (metric tons per capita)",
        "Renewables (% equivalent primary energy)", "GDP growth (annual %)",
        "GDP per capita", "Density (P/Km2)", "Land Area (Km2)",
        "Latitude", "Longitude"
    ]
    
    for i, col in enumerate(sample_columns, 1):
        st.write(f"{i}. {col}")
