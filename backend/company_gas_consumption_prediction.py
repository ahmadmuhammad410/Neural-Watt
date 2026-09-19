import pandas as pd
import numpy as np
import joblib
import os

def forecast_utility_pipeline(csv_path, model_path='../Models/Gas/Company/neural_watts_unified_engine.pkl', sample_building_id=None):
    print("=" * 70)
    print("           NEURAL WATTS - UTILITY INFERENCE ENGINE           ")
    print("=" * 70)
    
    # 1. تحميل الحزمة الموحدة بالكامل
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found.")
        return {}
        
    package = joblib.load(model_path)
    b2c_models = package['b2c']['models']
    b2c_features = package['b2c']['features']
    b2b_models = package['b2b']['models']
    b2b_features = package['b2b']['features']
    encoders = package['encoders']
    
    # 2. قراءة وتنظيف ملف الـ CSV
    raw_df = pd.read_csv(csv_path)
    
    if 'timestamp' not in raw_df.columns:
        raw_df.rename(columns={raw_df.columns[0]: 'timestamp'}, inplace=True)
        
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    
    if 'building_id' not in raw_df.columns:
        # تحويل من Wide Format إلى Long Format
        raw_df.set_index('timestamp', inplace=True)
        raw_df = raw_df.interpolate(method='time', limit=6).ffill().bfill()
        raw_df.reset_index(inplace=True)
        
        long_df = pd.melt(raw_df, id_vars=['timestamp'], var_name='building_id', value_name='gas_consumption')
        long_df['primary_use'] = long_df['building_id'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'unknown')
    else:
        long_df = raw_df.copy()
        if 'primary_use' not in long_df.columns:
            long_df['primary_use'] = long_df['building_id'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'unknown')
        
    # 3. التجميع اليومي
    daily_df = (
        long_df.groupby(['building_id', 'primary_use', pd.Grouper(key='timestamp', freq='D')])['gas_consumption']
        .sum()
        .reset_index()
    )
    daily_df.rename(columns={'gas_consumption': 'Daily_Gas'}, inplace=True)
    daily_df.sort_values(['building_id', 'timestamp'], inplace=True)
    
    # 4. استخراج الميزات المشتركة والـ Lags
    daily_df['sin_month'] = np.sin(2 * np.pi * daily_df['timestamp'].dt.month / 12.0)
    daily_df['cos_month'] = np.cos(2 * np.pi * daily_df['timestamp'].dt.month / 12.0)
    daily_df['sin_dayofweek'] = np.sin(2 * np.pi * daily_df['timestamp'].dt.dayofweek / 7.0)
    daily_df['cos_dayofweek'] = np.cos(2 * np.pi * daily_df['timestamp'].dt.dayofweek / 7.0)
    daily_df['quarter'] = daily_df['timestamp'].dt.quarter
    daily_df['Is_Weekend'] = daily_df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)

    grp = daily_df.groupby('building_id')['Daily_Gas']
    
    # ميزات الـ B2C (القصيرة)
    daily_df['Lag_1D'] = grp.shift(1)
    daily_df['Lag_2D'] = grp.shift(2)
    daily_df['Lag_7D'] = grp.shift(7)
    daily_df['Lag_14D'] = grp.shift(14)
    daily_df['Lag_30D'] = grp.shift(30)
    daily_df['Rolling_Mean_7D'] = grp.transform(lambda x: x.shift(1).rolling(7).mean())
    daily_df['Rolling_Mean_30D'] = grp.transform(lambda x: x.shift(1).rolling(30).mean())
    daily_df['Rolling_Std_7D'] = grp.transform(lambda x: x.shift(1).rolling(7).std())
    daily_df['diff_1d_7d'] = daily_df['Lag_1D'] - daily_df['Rolling_Mean_7D']
    daily_df['ratio_1d_7d'] = daily_df['Lag_1D'] / (daily_df['Rolling_Mean_7D'] + 1e-5)

    # ميزات الـ B2B (الطويلة)
    daily_df['Lag_90D'] = grp.shift(90)
    daily_df['Lag_180D'] = grp.shift(180)
    daily_df['Rolling_Mean_90D'] = grp.transform(lambda x: x.shift(1).rolling(90).mean())
    daily_df['Rolling_Mean_180D'] = grp.transform(lambda x: x.shift(1).rolling(180).mean())
    daily_df['Rolling_Std_30D'] = grp.transform(lambda x: x.shift(1).rolling(30).std())
    daily_df['ratio_30d_90d'] = daily_df['Rolling_Mean_30D'] / (daily_df['Rolling_Mean_90D'] + 1e-5)
    
    # سد البدايات
    eval_df = daily_df.bfill().ffill().copy()
    
    # حفظ اسم المبنى للعرض
    original_building_name = eval_df['building_id'].iloc[-1] if sample_building_id is None else sample_building_id
    
    # تشفير النصوص بأسماء الأعمدة الصحيحة
    eval_df['building_id_code'] = eval_df['building_id'].map(encoders.get('building_id', {})).fillna(0).astype(int)
    eval_df['primary_use_code'] = eval_df['primary_use'].map(encoders.get('primary_use', {})).fillna(0).astype(int)
    
    # 5. استخراج العينة المستهدفة
    if sample_building_id is not None:
        target_row = eval_df[eval_df['building_id'] == sample_building_id].iloc[-1:]
    else:
        target_row = eval_df.iloc[-1:]
        
    target_date = target_row['timestamp'].dt.date.values[0]
    
    # تجهيز مدخلات كل مسار بشكل منفصل
    for col in b2c_features:
        if col not in target_row.columns:
            target_row[col] = 0
            
    for col in b2b_features:
        if col not in target_row.columns:
            target_row[col] = 0
            
    input_b2c = target_row[b2c_features].astype(np.float32)
    input_b2b = target_row[b2b_features].astype(np.float32)

    # 6. توليد التوقعات
    print(f"\nTarget Building: {original_building_name}")
    print(f"Last Reading Date: {target_date}\n")
    print("-" * 65)

    forecast_results = {}

    # مسار المنازل (B2C)
    b2c_map = {
        'Target_Next_Day': '1. اليوم القادم (B2C)',
        'Target_Next_Week': '2. الأسبوع القادم (B2C)',
        'Target_Next_Month': '3. الشهر القادم (B2C)'
    }
    for target_key, label in b2c_map.items():
        val = float(b2c_models[target_key].predict(input_b2c)[0])
        forecast_results[target_key] = max(0.0, val)

    # مسار الشركات (B2B)
    b2b_map = {
        'Target_Next_Quarter': '4. الربع سنوي (B2B)',
        'Target_Next_SemiAnnual': '5. النصف سنوي (B2B)',
        'Target_Next_Annual': '6. السنوي الكامل (B2B)'
    }
    for target_key, label in b2b_map.items():
        val = float(b2b_models[target_key].predict(input_b2b)[0])
        if target_key == 'Target_Next_Annual' and val <= 0.0:
            semi_val = forecast_results.get('Target_Next_SemiAnnual', 0.0)
            quarter_val = forecast_results.get('Target_Next_Quarter', 0.0)
            val = semi_val * 2.0 if semi_val > 0 else quarter_val * 4.0

        forecast_results[target_key] = max(0.0, val)
        
    print("=" * 70)
    return forecast_results


def process_company_gas(df, facility_subtype, facility_size, holiday_usage, holiday_days, duration_months, data_handling_method, tmp_path=None):
    """
    Processes company gas consumption data.
    """
    if tmp_path and os.path.exists(tmp_path):
        csv_path = tmp_path
    else:
        # Fallback to the saved actual daily consumption
        csv_path = "Actual_daily_consumption.csv"
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
            
    try:
        forecast = forecast_utility_pipeline(csv_path)
    except Exception as e:
        print(f"Error in utility pipeline: {e}")
        forecast = {}

    def calc_waste(pred, factor=1.0):
        if not pred: return 0.0, 0.0
        gas_thresholds = {
            'Bakery': {'small': 2, 'medium': 6, 'large': 15},
            'Office': {'small': 0.1, 'medium': 0.5, 'large': 1.2},
            'Hotel': {'small': 1.5, 'medium': 6, 'large': 15},
            'Restaurant': {'small': 0.5, 'medium': 2, 'large': 4.5},
            'School': {'small': 0.3, 'medium': 1.5, 'large': 3},
            'SuperMarket': {'small': 0.3, 'medium': 1, 'large': 2},
        }
        base_threshold = 1000
        if facility_subtype in gas_thresholds:
            size = facility_size.lower() if facility_size else 'small'
            if size not in gas_thresholds[facility_subtype]:
                size = 'small'
            base_threshold = gas_thresholds[facility_subtype][size]
        
        limit = base_threshold * factor
        if pred > limit:
            waste_amt = pred - limit
            waste_pct = (waste_amt / limit) * 100
            return round(waste_amt, 2), round(waste_pct, 2)
        return 0.0, 0.0

    day_pred = forecast.get("Target_Next_Day")
    week_pred = forecast.get("Target_Next_Week")
    month_pred = forecast.get("Target_Next_Month")
    quarter_pred = forecast.get("Target_Next_Quarter")
    semi_annual_pred = forecast.get("Target_Next_SemiAnnual")
    annual_pred = forecast.get("Target_Next_Annual")

    day_pred = day_pred / 1000.0 if day_pred is not None else 0.0
    week_pred = week_pred / 1000.0 if week_pred is not None else 0.0
    month_pred = month_pred / 1000.0 if month_pred is not None else 0.0
    quarter_pred = quarter_pred / 1000.0 if quarter_pred is not None else 0.0
    semi_annual_pred = semi_annual_pred / 1000.0 if semi_annual_pred is not None else 0.0
    annual_pred = annual_pred / 1000.0 if annual_pred is not None else 0.0

    day_amt, day_pct = calc_waste(day_pred, factor=1/30)
    week_amt, week_pct = calc_waste(week_pred, factor=1/4)
    month_amt, month_pct = calc_waste(month_pred, factor=1.0)
    quarter_amt, quarter_pct = calc_waste(quarter_pred, factor=3.0)

    waste_dict = {
        "waste_day": day_amt,
        "waste_day_pct": day_pct,
        "waste_week": week_amt,
        "waste_week_pct": week_pct,
        "waste_month": month_amt,
        "waste_month_pct": month_pct,
        "waste_quarter": quarter_amt,
        "waste_quarter_pct": quarter_pct,
    }

    prediction_result = {
        "next_day": day_pred,
        "next_week": week_pred,
        "next_month": month_pred,
        "next_quarter": quarter_pred,
        "next_semi_annual": semi_annual_pred,
        "next_annual": annual_pred,
        "waste": waste_dict
    }
    
    return prediction_result
