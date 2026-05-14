import pandas as pd
import numpy as np

def generate_churn_dataset(n=2000, seed=42):
    np.random.seed(seed)

    age = np.random.randint(18, 75, n)
    gender = np.random.choice(['Male', 'Female'], n)
    tenure = np.random.randint(1, 72, n)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n,
                                p=[0.55, 0.25, 0.20])
    payment = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n)
    internet = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.20])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n)
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], n)
    support_calls = np.random.randint(0, 10, n)

    monthly = np.round(np.random.uniform(20, 120, n), 2)
    total = np.round(monthly * tenure * np.random.uniform(0.85, 1.15, n), 2)

    # Inject some missing values
    mask_total = np.random.rand(n) < 0.03
    total[mask_total] = np.nan
    mask_monthly = np.random.rand(n) < 0.01
    monthly[mask_monthly] = np.nan

    # Build churn probability based on features
    churn_prob = (
        0.30 * (contract == 'Month-to-month').astype(float) +
        0.10 * (internet == 'Fiber optic').astype(float) +
        0.15 * (tech_support == 'No').astype(float) +
        0.10 * (online_security == 'No').astype(float) +
        0.10 * (support_calls > 5).astype(float) +
        0.05 * (monthly > 80).astype(float) +
        0.15 * (tenure < 12).astype(float) -
        0.10 * (contract == 'Two year').astype(float)
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        'CustomerAge': age,
        'Gender': gender,
        'Tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'ContractType': contract,
        'PaymentMethod': payment,
        'InternetService': internet,
        'TechSupport': tech_support,
        'OnlineSecurity': online_security,
        'CustomerSupportCalls': support_calls,
        'Churn': churn
    })

    # Add a few duplicates
    dup_idx = np.random.choice(n, 30, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    return df


if __name__ == '__main__':
    df = generate_churn_dataset()
    df.to_csv('churn_dataset.csv', index=False)
    print(f"Dataset saved: {df.shape}")
