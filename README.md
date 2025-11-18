# 🏦 Credit Scoring System: M-Pesa Transaction-Based Loan Assessment

A complete end-to-end ML system that predicts loan default risk using M-Pesa mobile money transaction patterns. The project includes synthetic data generation, feature engineering, model training with comparison, and production-ready API deployment.

**🔗 Live Demo:** [https://credit-scoring-0jwt.onrender.com](https://your-deployment-url.onrender.com)

---

## 📊 Project Overview

This system predicts whether a loan applicant will repay or default based on:
- **M-Pesa transaction patterns** (income, expenses, Fuliza usage)
- **Demographics** (age, employment type, education)
- **Financial history** (past loans, defaults, credit score)

The project demonstrates a complete ML pipeline: from synthetic data generation to production deployment with FastAPI.

### Why This Matters
In Kenya and similar markets, millions lack traditional credit history but have rich mobile money data. This system enables credit scoring for the underbanked.

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10 |
| **ML & Data** | scikit-learn, XGBoost, LightGBM, pandas, NumPy |
| **API Framework** | FastAPI, Uvicorn |
| **Validation** | Pydantic |
| **Config** | YAML |
| **Deployment** | Render |
| **Others** | Joblib (persistence), Custom logging & exceptions |

---

## 🎯 Key Features

### 1. Synthetic Data Generation
Since real M-Pesa data is unavailable, we generated realistic transaction patterns:
- **1,000 applicant profiles** with demographics and financial history
- **~850,000 transactions** (850 per applicant over 12 months)
- **Employment-based patterns**: Salaried (stable), Self-employed (variable), Informal (irregular)
- **Realistic Fuliza usage**: Frequency correlates with income stability

### 2. Feature Engineering (24 Features)
Extracted from raw transactions:
- **Income**: Average, volatility, trend, sources
- **Cash Flow**: Net flow, negative months, minimum balance
- **Debt**: Fuliza frequency, repayment rate, debt-to-income ratio
- **Behavior**: Transaction frequency, savings rate, expense consistency

### 3. Model Training & Selection
Trained 4 models with cross-validation:

| Model | Test Accuracy | Test F1 | Test ROC-AUC |
|-------|---------------|---------|--------------|
| **LightGBM** ⭐ | **96.0%** | **97.3%** | **99.3%** |
| XGBoost | 96.0% | 97.3% | 99.0% |
| Logistic Regression | 95.0% | 96.6% | 98.4% |
| Random Forest | 95.0% | 96.6% | 98.3% |

**Winner: LightGBM** - Best F1 and ROC-AUC, minimal overfitting

### 4. Production API
- **FastAPI** with automatic validation (Pydantic)
- **Interactive docs** at `/docs` (Swagger UI)
- **Web interface** for quick testing
- **Two prediction modes**: Raw data (full pipeline) or pre-computed features

---

## 🧠 ML Pipeline Workflow

```
1. Data Generation → synthetic_data/
   Generate realistic M-Pesa transaction patterns

2. Feature Engineering → src/data/
   Extract 24 features from transactions

3. Model Training → src/components/
   Train & compare 4 models with CV

4. Model Selection → src/pipeline/
   Select best model (LightGBM)

5. API Deployment → app.py
   Serve predictions via FastAPI
```

---

## 🚀 Getting Started

### 1️⃣ Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd credit-scoring

# Create virtual environment
conda create -n credit-scoring python=3.10
conda activate credit-scoring

# Install dependencies
pip install -r requirements.txt

# Install project
pip install -e .
```

### 2️⃣ Train Model

```bash
# Run training pipeline
python -m src.pipeline.train_pipeline
```

**What happens:**
- Loads synthetic data (demographics, financial history, transactions)
- Extracts 24 features per applicant
- Trains 4 models with 5-fold cross-validation
- Saves best model to `models/best_model.pkl`

**Outputs:**
- `models/best_model.pkl` - Trained LightGBM model
- `models/scaler.pkl` - Feature scaler
- `models/model_comparison.csv` - Performance metrics

**Training time:** ~2-3 minutes for 1,000 applicants

### 3️⃣ Make Predictions

**Option A: Command-line**
```bash
python -m src.pipeline.predict_pipeline --applicant-id 1
```

**Option B: Start API server**
```bash
python app.py
```

Then visit:
- **Web Interface:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## 📂 Project Structure

```
credit-scoring/
│
├── app.py                          # FastAPI application
├── Procfile                        # Render deployment
├── requirements.txt                # Dependencies
├── setup.py                        # Package setup
│
├── configs/
│   └── config.yaml                 # Project configuration
│
├── data/
│   ├── raw/                        # Synthetic data (CSV)
│   └── processed/                  # Engineered features
│
├── models/                         # Saved models & artifacts
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_comparison.csv
│
├── src/
│   ├── components/                 # Core ML components
│   │   ├── data_ingestion.py      # Load CSV data
│   │   ├── data_transformation.py # Feature engineering
│   │   └── model_trainer.py       # Train & evaluate models
│   │
│   ├── pipeline/                   # ML pipelines
│   │   ├── train_pipeline.py      # Training workflow
│   │   └── predict_pipeline.py    # Prediction workflow
│   │
│   ├── data/                       # Data utilities
│   │   └── feature_extractor.py   # Extract features from transactions
│   │
│   └── utils/                      # Utilities
│       ├── logger.py               # Logging setup
│       ├── exception.py            # Custom exceptions
│       └── config_loader.py        # YAML config reader
│
├── scripts/                        # Executable scripts
│   ├── train.py                    # Train models (CLI)
│   └── predict.py                  # Make predictions (CLI)
│
├── templates/                      # Web frontend
│   └── index.html                  # Prediction form
│
└── notebooks/                      # Jupyter notebooks
    └── EDA.ipynb                   # (Optional) Data exploration
```

---

## 📊 Models Compared

### Training Configuration
- **Dataset:** 1,000 applicants (800 train, 200 test)
- **Features:** 24 engineered features
- **Target:** Binary (0=Default, 1=Repaid)
- **Class Distribution:** 75% Repaid, 25% Default
- **Validation:** 5-fold cross-validation
- **Metric:** F1-score (balances precision & recall)

### Performance Comparison

| Model | CV F1 | Train Acc | Test Acc | Test F1 | ROC-AUC |
|-------|-------|-----------|----------|---------|---------|
| **LightGBM** | 0.961±0.005 | 100.0% | **96.0%** | **97.3%** | **99.3%** |
| XGBoost | 0.960±0.008 | 99.9% | 96.0% | 97.3% | 99.0% |
| Logistic Regression | 0.968±0.006 | 95.5% | 95.0% | 96.6% | 98.4% |
| Random Forest | 0.965±0.006 | 99.4% | 95.0% | 96.6% | 98.3% |

**Why LightGBM wins:**
- Highest test F1 and ROC-AUC
- No overfitting (100% train → 96% test is acceptable for tree models)
- Consistent cross-validation performance
- Fast inference

---

## 🎯 Feature Groups

### Transaction Features (15)
**Income (4 features):**
- `monthly_avg_income` - Average monthly earnings
- `income_volatility_cv` - Income stability (lower = better)
- `income_trend_6months` - Recent income growth
- `num_income_sources` - Income diversity

**Cash Flow (4 features):**
- `monthly_net_cashflow` - Income minus expenses
- `months_negative_cashflow` - Deficit months (red flag)
- `balance_min` - Lowest balance reached
- `months_balance_below_500` - Financial stress indicator

**Debt (4 features):**
- `fuliza_frequency` - Overdraft usage frequency
- `fuliza_repayment_rate` - % of overdrafts repaid
- `debt_to_income_ratio` - Debt burden
- `avg_fuliza_amount` - Typical overdraft size

**Behavior (3 features):**
- `transaction_frequency` - Account activity
- `savings_rate` - % of income saved
- `expense_consistency_cv` - Spending predictability

### Demographic Features (4)
- `age` - Applicant age
- `years_employed` - Employment duration
- `education_level_encoded` - 0=Primary, 1=Secondary, 2=Tertiary
- `employment_type` - One-hot encoded (Salaried, Self-employed, Informal)

### Financial History Features (5)
- `account_age_months` - M-Pesa account age
- `past_defaults_count` - Previous loan defaults
- `past_repaid_loans` - Successfully repaid loans
- `credit_score` - Traditional credit score (if available)
- `overdraft_history` - Past overdraft occurrences

---

## 🌐 API Endpoints

### 1. Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-11-18T10:30:00",
  "version": "1.0.0"
}
```

### 2. Predict (Full Pipeline)
```bash
POST /predict
Content-Type: application/json
```
**Request:**
```json
{
  "demographics": {
    "age": 35,
    "employment_type": "Salaried",
    "years_employed": 8,
    "education_level": "Secondary"
  },
  "financial_history": {
    "account_age_months": 48,
    "past_defaults_count": 0,
    "past_repaid_loans": 3,
    "credit_score": 680,
    "overdraft_history": 2
  },
  "transactions": [{
    "date": "2024-01-01",
    "transaction_type": "Income",
    "amount": 50000,
    "balance": 50000
  }]
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Repaid",
  "probability_default": 0.024,
  "probability_repay": 0.976,
  "risk_level": "Low",
  "recommendation": "Approve - High confidence",
  "timestamp": "2024-11-18T10:35:20"
}
```

### 3. Interactive Documentation
```
GET /docs
```
Swagger UI with interactive API testing.

---

## 🚢 Deployment (Render)

### Quick Deploy

1. **Push to GitHub:**
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

2. **Deploy on Render:**
- Go to [render.com](https://render.com)
- New → Web Service
- Connect repository
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Click "Create Web Service"

3. **Wait 5-10 minutes** for deployment

4. **Access your API:**
```
https://your-app.onrender.com
https://your-app.onrender.com/docs
```

### Environment Variables (Optional)
For production, set:
- `PORT` - Automatic on Render
- `PYTHON_VERSION` - 3.10

---

## 📈 Sample Results

### Example Prediction: Low-Risk Applicant
```
Demographics: 35 years old, Salaried, 8 years employed
Financial History: 0 defaults, 3 repaid loans, credit score 680
Transaction Pattern: Stable monthly income, low Fuliza usage

Prediction: REPAID (97.6% confidence)
Risk Level: Low
Recommendation: Approve - High confidence
```

### Example Prediction: High-Risk Applicant
```
Demographics: 28 years old, Informal, 2 years employed
Financial History: 2 defaults, 1 repaid loan, no credit score
Transaction Pattern: Irregular income, frequent Fuliza usage

Prediction: DEFAULT (82.3% confidence)
Risk Level: High
Recommendation: Reject - High default risk
```

---

**Test API locally:**
```bash
python app.py
# Visit http://localhost:8000
```

---

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Omido Benard**

---

## 🙏 Acknowledgments

- Inspired by M-Pesa's impact on financial inclusion in Kenya
- Built as a demonstration of end-to-end ML engineering
- Synthetic data generation approach adapted from industry best practices

---

**⭐ If this project helped you, please star the repository!**
