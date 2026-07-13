# Superstore Sales Analytics Dashboard

A 4-page Streamlit app built on the Superstore `train.csv` dataset:

1. **Sales Overview** — total sales by year, monthly trend, region × category with filters
2. **Forecast Explorer** — XGBoost forecast (best model from the notebook) by Category/Region, 1–3 months ahead, with MAE/RMSE
3. **Anomaly Report** — Isolation Forest anomaly detection on weekly sales
4. **Product Demand Segments** — KMeans clustering of sub-categories into demand segments

## File structure
```
.
├── streamlit_app.py           # Home page (entry point)
├── utils.py                   # Shared data loading + model logic (cached)
├── train.csv                  # Dataset (must sit next to streamlit_app.py)
├── requirements.txt
└── pages/
    ├── 1_Sales_Overview.py
    ├── 2_Forecast_Explorer.py
    ├── 3_Anomaly_Report.py
    └── 4_Product_Demand_Segments.py
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
The app opens at `http://localhost:8501`. Use the sidebar to switch pages.

## Deploy to Streamlit Community Cloud (free)

1. **Push this folder to a public GitHub repo** (keep the structure above exactly as-is —
   `train.csv` must be committed alongside `streamlit_app.py` so the app can read it in the cloud).
   ```bash
   git init
   git add .
   git commit -m "Superstore sales dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. Go to **https://share.streamlit.io** and sign in with your GitHub account.

3. Click **"New app"**, then select:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`

4. Click **"Deploy"**. First build takes 2–5 minutes (installing XGBoost, scikit-learn, etc.).

5. Once live, you'll get a public URL like:
   `https://<your-app-name>.streamlit.app`
   Submit that link.

### Notes
- `train.csv` is ~2 MB, well under Streamlit Cloud's repo size limits — no Git LFS needed.
- If you'd rather not commit the raw CSV to a public repo, keep the repo **private**
  (Streamlit Community Cloud supports deploying private repos to a public app URL) or
  swap `pd.read_csv("train.csv")` in `utils.py` for a load from cloud storage.
- All expensive computations (data load, model training, clustering) are wrapped in
  `@st.cache_data`, so switching between pages/filters after the first run is fast.
