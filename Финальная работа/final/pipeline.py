import datetime
import dill
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def add_custom_features(df):
    df_out = df.copy()

    organic_mediums = ["organic", "referral", "(none)"]
    df_out["is_organic"] = df_out["utm_medium"].isin(organic_mediums).astype(int)

    top_cities = ['Moscow', 'Saint Petersburg', 'Ekaterinburg']
    df_out["is_major_city"] = df_out["geo_city"].isin(top_cities).astype(int)

    df_out["clicks_per_visit"] = df_out["hit_number"] / (
        df_out["visit_number"] + 1e-5
    )
    cols_to_drop = [
        "session_id", "visit_date", "hit_date", "geo_city",
        "utm_source", "utm_campaign", "utm_adcontent", "utm_keyword",
        "device_brand", "device_model", "device_screen_resolution",
        "hit_page_path", "event_category", "event_label"
    ]
    df_out = df_out.drop(columns=cols_to_drop, errors="ignore")

    return df_out

def main():
    print("1. Загрузка данных")
    df_sessions = pd.read_pickle("ga_sessions.pkl")
    df_hits = pd.read_pickle("ga_hits-002.pkl")

    print("2. Подготовка целевой переменной (Таргет)")
    target_actions = [
        "sub_car_claim_click",
        "sub_car_claim_submit_click",
        "sub_open_dialog_click",
        "sub_custom_question_submit_click",
        "sub_call_number_click",
        "sub_callback_submit_click",
        "sub_submit_success",
        "sub_car_request_submit_click",
    ]
    df_hits["is_target"] = (
        df_hits["event_action"].isin(target_actions).astype(int)
    )

    df_hits_grouped = (
        df_hits.groupby("session_id")
        .agg({"is_target": "max", "hit_number": "max"})
        .reset_index()
    )

    print("3. Первичное объединение таблиц")
    df_merged = pd.merge(df_sessions, df_hits_grouped, on="session_id", how="left")
    df_merged["is_target"] = df_merged["is_target"].fillna(0).astype(int)
    df_merged["hit_number"] = df_merged["hit_number"].fillna(0)
    text_cols = ["utm_medium", "device_category", "device_os", "device_browser", "geo_city"]
    for col in text_cols:
        df_merged[col] = df_merged[col].fillna("unknown").astype(str)
    df_merged = df_merged.drop_duplicates()

    categorical_features = [
        "utm_medium",
        "device_category",
        "device_os",
        "device_browser",
    ]

    feature_cols = categorical_features + ["visit_number", "hit_number", "geo_city"]
    X = df_merged[feature_cols].copy()
    y = df_merged["is_target"]

    print("4. Настройка трансформеров колонок")
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer,
         ['visit_number', 'hit_number', 'is_organic', 'is_major_city', 'clicks_per_visit']),
        ('categorical', categorical_transformer, categorical_features)
    ])

    print("5. Сборка сквозного блока трансформации")
    transform_pipeline = Pipeline(steps=[
        ('features', FunctionTransformer(add_custom_features)),
        ('transformation', preprocessor)
    ])

    models = [
        LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42),
        RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1)
    ]

    best_score = 0.0
    best_pipe = None

    for model in models:
        pipe = Pipeline(steps=[
            ('transform', transform_pipeline),
            ('classifier', model)
        ])

        print(f"Валидация модели {type(model).__name__}...")

        score = cross_val_score(pipe, X, y, cv=3, scoring='roc_auc', n_jobs=-1)
        print(f'model: {type(model).__name__}, ROC-AUC mean: {score.mean():.4f}, std: {score.std():.4f}\n')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    classifier_name = type(best_pipe.named_steps['classifier']).__name__
    print(f'=== BEST MODEL: {classifier_name}, ROC-AUC: {best_score:.4f} ===')

    print("7. Обучение финального пайплайна на всех данных")
    best_pipe.fit(X, y)

    print("8. Сохранение пайплайн в sber_pipe.pkl")
    with open('sber_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'sber prediction pipeline',
                'author': 'Anton Savinov',
                'version': '1.0',
                'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': classifier_name,
                'roc_auc': best_score
            }
        }, file)
    print("Пайплайн успешно сохранен через dill!")



if __name__ == "__main__":
    main()