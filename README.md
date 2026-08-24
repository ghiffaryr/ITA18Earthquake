# From Data to Decision: Physics-Constrained Machine Learning for Seismic Risk Assessment in Italy

A comprehensive machine learning analysis project for predicting ground motion in Italian earthquakes using the ITA18 seismic dataset.

## Overview

This project implements and evaluates multiple machine learning models to predict ground motion parameters for Italian earthquakes. It combines seismic data analysis, feature engineering, and model evaluation techniques to build robust predictive models for earthquake ground motion.

## Project Structure

```
Physics-Constrained-Machine-Learning-for-Seismic-Risk-Assessment-in-Italy/
├── data/                          # Dataset and metadata
│   ├── ITA18_SA_flatfile.csv     # Main earthquake dataset
│   ├── ec8_code_method.txt       # Eurocode 8 classification codes
│   ├── fm_type_code.txt          # Focal mechanism type codes
│   ├── housing_code.txt          # Building type codes
│   ├── installation_code.txt     # Sensor installation codes
│   ├── instrument_type_code.txt  # Instrument type codes
│   ├── net_code.txt              # Network identifier codes
│   ├── proximity_code.txt        # Proximity classification codes
│   ├── vs30_calc_method.txt      # Vs30 calculation method codes
│   ├── vs30_meas_type.txt        # Vs30 measurement type codes
│   ├── web_availability_code.txt # Web availability codes
│   └── ref.txt                   # Reference bibliography
├── src/
│   └── Code.ipynb               # Main analysis notebook
├── models/                       # Trained model artifacts
│   ├── catboost_model.cbm       # CatBoost model
│   ├── lgbm_model.txt           # LightGBM model
│   ├── xgb_model.json           # XGBoost model
│   ├── robust_scaler.joblib     # Feature scaler
│   └── pytorch_model/
│       └── model.pt             # PyTorch neural network model
├── figures/                      # Generated visualizations
│   ├── deployment/              # Deployment analysis plots
│   ├── eda/                      # Exploratory data analysis plots
│   └── model_diagnostics/       # Model evaluation plots
├── reports/                      # Documentation
│   └── Presentation.tex         # LaTeX presentation
└── papers/                       # Research papers and references
```

## Dataset

The project uses the **ITA18 earthquake dataset** from the Italian National Institute of Geophysics and Volcanology (INGV). The dataset includes:

- **Ground motion records**: Spectral accelerations and other ground motion parameters
- **Earthquake parameters**: Magnitude, focal depth, focal mechanism
- **Site characteristics**: Vs30 values, soil classifications, geotechnical parameters
- **Source and path properties**: Distance metrics, rupture parameters
- **Metadata**: Instrument types, network information, installation details

### Data Files

- `ITA18_SA_flatfile.csv` - Main flatfile containing earthquake records with ground motion parameters
- `ref.txt` - Comprehensive bibliography with data sources and references

## Analysis Pipeline

The project follows a structured machine learning pipeline:

### 1. **Phase 1: Setup & Data Loading**

- Environment initialization
- Library imports
- Data loading and validation

### 2. **Phase 2: Data Understanding & EDA**

- Statistical summaries
- Distribution analysis
- Missing value assessment
- Correlation analysis
- Visualizations saved to `figures/eda/`

### 3. **Phase 3: Data Preparation**

- Feature engineering
- Missing value handling
- Data scaling and normalization
- Train-test split

### 4. **Phase 4: Modeling**

- Multiple model implementations:
  - **XGBoost** - Gradient boosting
  - **LightGBM** - Light gradient boosting
  - **CatBoost** - Categorical boosting
  - **PyTorch Neural Network** - Deep learning
- Hyperparameter tuning
- Cross-validation

### 5. **Phase 5: Model Evaluation**

- Performance metrics (R², RMSE, MAE)
- Residual analysis
- Feature importance
- Prediction error distribution
- Model comparison

### 6. **Phase 6: Deployment**

- Model serialization
- Scalability assessment
- Production-ready artifacts

## Models

The project trains and evaluates four different model types:

| Model          | File                     | Type                 | Framework |
| -------------- | ------------------------ | -------------------- | --------- |
| XGBoost        | `xgb_model.json`         | Gradient Boosting    | XGBoost   |
| LightGBM       | `lgbm_model.txt`         | Gradient Boosting    | LightGBM  |
| CatBoost       | `catboost_model.cbm`     | Categorical Boosting | CatBoost  |
| Neural Network | `pytorch_model/model.pt` | Deep Learning        | PyTorch   |

## Key Features

- **Multiple ML Frameworks**: Comparison of XGBoost, LightGBM, CatBoost, and PyTorch models
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Feature Importance Analysis**: Understanding which parameters drive predictions
- **Model Diagnostics**: Residual plots, error analysis, and performance metrics
- **Scalable Architecture**: Models saved and ready for deployment

## Results & Outputs

### Figures Generated

- **EDA Plots**: Data distributions, correlations, time series patterns
- **Model Diagnostics**: Residual plots, predicted vs. actual, error histograms
- **Deployment Analysis**: Model performance across different subsets

### Training Information

- `catboost_info/` directory contains CatBoost training logs:
  - `catboost_training.json` - Full training metrics
  - `learn_error.tsv` - Learning error tracking
  - `time_left.tsv` - Training time estimates

## Technologies Used

- **Data Processing**: pandas, NumPy
- **Machine Learning**: XGBoost, LightGBM, CatBoost, scikit-learn
- **Deep Learning**: PyTorch
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Preprocessing**: scikit-learn (RobustScaler)

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone or download the project
cd Physics-Constrained-Machine-Learning-for-Seismic-Risk-Assessment-in-Italy

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn
pip install xgboost lightgbm catboost
pip install torch
```

## Usage

### Running the Analysis

```bash
# Navigate to the src directory
cd src

# Open the Jupyter notebook
jupyter notebook Code.ipynb
```

The notebook is organized into logical sections that can be run sequentially:

1. Run setup cells to initialize environment
2. Execute data loading cells
3. Run EDA sections to explore data
4. Execute modeling cells to train models
5. Review model evaluation results

### Using Trained Models

```python
import joblib
import json

# Load scaler
scaler = joblib.load('models/robust_scaler.joblib')

# Load XGBoost model
import xgboost as xgb
xgb_model = xgb.Booster()
xgb_model.load_model('models/xgb_model.json')

# Scale your data and make predictions
X_scaled = scaler.transform(X)
predictions = xgb_model.predict(X_scaled)
```

## Data Citation

The ITA18 dataset is provided by the Italian National Institute of Geophysics and Volcanology (INGV). For detailed references and data sources, see `data/ref.txt`.

## References

Key earthquake catalogs and databases included:

- **CPTI11**: Parametric Catalogue of Italian Earthquakes
- **CSI**: Catalogo della sismicità italiana
- **DISS**: Database of Individual Seismogenic Sources
- **Global CMT**: Centroid Moment Tensor catalog
- **ISC**: International Seismological Centre

For comprehensive bibliography, refer to `data/ref.txt`.

## Key Insights

- Ground motion predictions depend on multiple earthquake and site parameters
- Machine learning models effectively capture non-linear relationships
- Ensemble methods (Gradient Boosting) show strong performance
- Feature importance analysis reveals critical predictor variables
- Model evaluation includes rigorous cross-validation and diagnostic plots

## Project Author

**Ghiffary Rifqialdi**

## License

Please refer to the data sources and the ITA18 project for licensing information.

## Notes

- All model artifacts are serialized and ready for deployment
- Training logs are available in `catboost_info/` for reproducibility
- Visualizations are saved during notebook execution
- The project uses RobustScaler for feature normalization to handle outliers

## Future Improvements

- Real-time prediction API
- Model interpretability analysis (SHAP values)
- Additional ground motion parameters (PGA, PGV)
- Regional-specific model variants
- Uncertainty quantification

---

**Last Updated**: August 24, 2026
