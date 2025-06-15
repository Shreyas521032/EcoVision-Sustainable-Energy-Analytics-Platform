# 🌍 EcoVision: Sustainable Energy Analytics Platform

An advanced AI/ML-powered analytics platform for exploring global sustainable energy trends, predictions, and insights from comprehensive data spanning 2000-2020.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecovision-sustainable-energy-analytics-platform-ssmp.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

EcoVision Analytics is a comprehensive sustainability analytics dashboard that transforms complex energy data into actionable insights. Built as an AI/ML project, it combines data science, machine learning, and interactive visualization to help understand global energy transition patterns and predict future trends.

### 🔥 Key Features

- 📊 **Interactive Data Visualization** - Dynamic charts, maps, and graphs
- 🌍 **Geographic Analysis** - World maps with sustainability metrics
- 📈 **Trend Analysis** - Time series analysis with growth rate calculations
- 🔄 **Correlation Studies** - Advanced statistical relationship analysis
- 🤖 **Machine Learning Models** - Predictive modeling with multiple algorithms
- 🔮 **Time Series Forecasting** - Future trend predictions with scenario analysis
- 🎛️ **Interactive Filters** - Country and year-based data filtering
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile

## 🚀 Live Project

**[🌐 Live Deployed Project](https://ecovision-sustainable-energy-analytics-platform-ssmp.streamlit.app/)**

## 📊 Dashboard Pages

### 1. 📊 Overview & Key Metrics
- Dataset summary and statistics
- Top performing countries analysis
- Key sustainability indicators
- Real-time data insights

### 2. 🌍 Geographic Analysis
- Interactive world maps
- Regional comparisons
- Geographic distribution of sustainability metrics
- Country-specific insights

### 3. 📈 Trend Analysis
- Global trend visualization
- Country-specific comparisons
- Growth rate analysis
- Multi-indicator trend tracking

### 4. 🔄 Correlation Analysis
- Correlation matrix heatmaps
- Scatter plot analysis
- Statistical relationship discovery
- Key correlation insights

### 5. 🤖 Predictive Modeling
- Linear Regression models
- Random Forest algorithms
- Model performance comparison
- Feature importance analysis
- Prediction vs actual value visualization

### 6. 🔮 Time Series Forecasting
- Linear and polynomial trend forecasting
- Multi-country comparative forecasts
- Scenario analysis (optimistic/realistic/pessimistic)
- Future trend predictions (1-10 years)

## 🛠️ Technologies Used

### Core Framework
- **Streamlit** - Web application framework
- **Python 3.8+** - Programming language

### Data Science & ML
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualization

### Interactive Visualization
- **Plotly** - Interactive charts and maps
- **Plotly Express** - High-level plotting interface

### Additional Libraries
- **Requests** - HTTP library for data fetching
- **Warnings** - Warning control

## 📥 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ecovision-sustainable-energy-analytics-platform.git
cd ecovision-sustainable-energy-analytics-platform
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Open in Browser
The app will automatically open in your default browser at `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.15.0
seaborn>=0.11.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
requests>=2.28.0
```

## 📊 Dataset Information

### Data Source
- **Dataset**: Global Data on Sustainable Energy (2000-2020)
- **Coverage**: 200+ countries and territories
- **Time Period**: 21 years (2000-2020)
- **Format**: CSV file

### Key Indicators
- 🔌 **Energy Access**: Electricity access, clean cooking fuels
- 🌱 **Renewable Energy**: Share in consumption, capacity per capita
- ⚡ **Energy Sources**: Fossil fuels, nuclear, renewables (TWh)
- 🌿 **Environmental**: CO2 emissions, low-carbon electricity
- 💰 **Economic**: GDP growth, energy intensity, financial flows
- 📍 **Geographic**: Country coordinates for mapping

### Expected Columns
```
Entity, Year, Access to electricity (% of population),
Access to clean fuels for cooking (% of population),
Renewable-electricity-generating-capacity-per-capita,
Financial flows to developing countries (US $),
Renewable energy share in total final energy consumption (%),
Electricity from fossil fuels (TWh),
Electricity from nuclear (TWh),
Electricity from renewables (TWh),
Low-carbon electricity (% electricity),
Primary energy consumption per capita (kWh/person),
Energy intensity level of primary energy (MJ/$2011 PPP GDP),
Value_co2_emissions (metric tons per capita),
Renewables (% equivalent primary energy),
GDP growth (annual %), GDP per capita,
Density (P/Km2), Land Area (Km2),
Latitude, Longitude
```

## 🚀 Usage Guide

### 1. Upload Dataset
- Click "Browse files" and upload your CSV dataset
- Or click "🔄 Load Sample Data" to use the repository dataset
- Or download directly using the "📥 Download Dataset CSV" button

### 2. Navigate Pages
Use the sidebar navigation to explore different analysis types:
- Start with **Overview** for general insights
- Use **Geographic Analysis** for spatial patterns
- Explore **Trends** for temporal analysis
- Check **Correlations** for relationships
- Try **Predictive Modeling** for ML insights
- Use **Forecasting** for future predictions

### 3. Apply Filters
- **Year Range**: Slider to focus on specific time periods
- **Countries**: Multi-select for country-specific analysis
- **All filters update visualizations dynamically**

### 4. Interact with Charts
- **Hover** for detailed information
- **Zoom** and **pan** on maps and charts
- **Download** charts as images
- **Expand** sections for detailed data

## 🎨 Features in Detail

### Interactive Visualizations
- **Bar Charts**: Top performers, comparisons
- **Line Charts**: Trends over time, growth rates
- **Scatter Plots**: Correlations, relationships
- **Heatmaps**: Correlation matrices
- **World Maps**: Geographic distributions
- **Subplots**: Multi-indicator analysis

### Machine Learning Capabilities
- **Regression Models**: Linear and Random Forest
- **Performance Metrics**: MAE, MSE, R²
- **Feature Importance**: Variable significance
- **Train/Test Split**: Temporal validation
- **Prediction Visualization**: Actual vs predicted

### Forecasting Features
- **Linear Trends**: Simple trend extrapolation
- **Polynomial Trends**: Non-linear patterns
- **Scenario Analysis**: Multiple growth rate scenarios
- **Comparative Forecasts**: Multi-country predictions
- **Confidence Intervals**: Uncertainty quantification

## 📁 Project Structure

```
ecovision-sustainable-energy-analytics-platform/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── Dataset/
│   └── global-data-on-sustainable-energy.csv
└── .gitignore                     # Git ignore file
```

## 🔧 Configuration

### Customization Options
- **Color Schemes**: Modify color palettes in the code
- **Chart Types**: Add new visualization types
- **Indicators**: Include additional sustainability metrics
- **Forecasting Models**: Implement advanced ML forecasting
- **Regional Groupings**: Customize geographic regions

### Performance Optimization
- **Caching**: Built-in Streamlit caching for data processing
- **Lazy Loading**: Efficient data loading strategies
- **Memory Management**: Optimized for large datasets

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository and deploy
5. Your app will be live at `https://your-app-name.streamlit.app`

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 1. Fork the Repository
```bash
git clone https://github.com/your-username/ecovision-sustainable-energy-analytics-platform.git
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/amazing-feature
```

### 3. Make Changes
- Add new features
- Fix bugs
- Improve documentation
- Enhance visualizations

### 4. Submit Pull Request
```bash
git commit -m 'Add amazing feature'
git push origin feature/amazing-feature
```

## 📈 Use Cases

### Academic Research
- **Energy Policy Analysis**: Compare policies across countries
- **Climate Change Studies**: Analyze emission trends
- **Economic Research**: Study energy-economy relationships

### Business Intelligence
- **Investment Decisions**: Identify renewable energy opportunities
- **Market Analysis**: Understand energy market trends
- **Risk Assessment**: Evaluate sustainability risks

### Government & Policy
- **Policy Planning**: Evidence-based policy development
- **Progress Tracking**: Monitor SDG 7 progress
- **International Comparisons**: Benchmark against other nations

### Educational Purposes
- **Data Science Learning**: Hands-on analytics experience
- **Sustainability Education**: Interactive learning tool
- **Research Training**: Advanced analysis techniques

## 🏆 Project Achievements

### Technical Accomplishments
- ✅ **Comprehensive Analytics**: 6 major analysis modules
- ✅ **Machine Learning Integration**: Multiple ML algorithms
- ✅ **Interactive Visualizations**: 20+ chart types
- ✅ **Real-time Filtering**: Dynamic data exploration
- ✅ **Forecasting Capabilities**: Future trend predictions
- ✅ **Geographic Analysis**: World map integrations

### AI/ML Features
- ✅ **Predictive Modeling**: Regression algorithms
- ✅ **Feature Engineering**: Data preprocessing
- ✅ **Model Evaluation**: Performance metrics
- ✅ **Time Series Analysis**: Trend forecasting
- ✅ **Statistical Analysis**: Correlation studies

### Get Help
- 📧 **Email**: shreyas200410@gmail.com

---

**🌱 Built with passion for a sustainable future 🌱**

**Made with ❤️ by Shreyas for the 🌍**
