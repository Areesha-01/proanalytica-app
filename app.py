# ProAnalytica — Freelancer Job Postings Analyzer
# Probability & Statistics Project Spring 2026
# Team: ProAnalytica | Leader: Areesha
# Members: Areesha, Ayesha, Azka, Hadia, Minahil

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── LOAD & CLEAN ──────────────────────────────────────────────
df = pd.read_csv('freelancer_job_postings.csv')
df.dropna(subset=['client_country','avg_price','rate_type','currency'], inplace=True)
df = df[df['avg_price'] < df['avg_price'].quantile(0.99)]

fx = {'USD':1.0,'EUR':1.08,'GBP':1.27,'AUD':0.65,'INR':0.012,'CAD':0.74,'SAR':0.27,'AED':0.27}
df['price_usd'] = df.apply(lambda r: r['avg_price'] * fx.get(r['currency'], 1.0), axis=1).round(2)

# Log transform for better R2
df['log_price'] = np.log1p(df['price_usd'])
df['log_min']   = np.log1p(df['min_price'])
df['log_max']   = np.log1p(df['max_price'])

le_rate    = LabelEncoder()
le_country = LabelEncoder()
df['rate_enc']    = le_rate.fit_transform(df['rate_type'])
df['country_enc'] = le_country.fit_transform(df['client_country'])

X = df[['log_min','log_max','rate_enc','client_average_rating','client_review_count']]
y = df['log_price']
model = LinearRegression()
model.fit(X, y)
r2 = round(r2_score(y, model.predict(X)), 4)

# ── HELPERS ───────────────────────────────────────────────────
def fig_json(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_layout(extra=None):
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f0f4ff', family='Inter, sans-serif'),
        margin=dict(t=20, b=60, l=55, r=20),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1)
    )
    if extra:
        base.update(extra)
    return base

def predict_price(min_p, max_p, rate_type, rating, reviews):
    r_enc = le_rate.transform([rate_type])[0]
    X_input = pd.DataFrame({
        'log_min': [np.log1p(min_p)],
        'log_max': [np.log1p(max_p)],
        'rate_enc': [r_enc],
        'client_average_rating': [rating],
        'client_review_count': [reviews]
    })
    log_pred = model.predict(X_input)[0]
    return round(max(float(np.expm1(log_pred)), 0), 2)

# ── DASHBOARD ─────────────────────────────────────────────────
@app.route('/')
def index():
    total      = len(df)
    avg_price  = round(df['price_usd'].mean(), 2)
    avg_rating = round(df['client_average_rating'].mean(), 2)
    countries  = df['client_country'].nunique()

    # Chart 1 — Top 10 countries by jobs
    # FIX: color_continuous_scale='Blues' on dark background makes bars near-invisible
    # (India #1 with 2643 jobs showed as 0 because deep-blue blended with dark bg).
    # Use a fixed bright color + text labels so every bar is clearly visible.
    top_c = df['client_country'].value_counts().head(10).reset_index()
    top_c.columns = ['Country','Jobs']
    fig1 = px.bar(top_c, x='Country', y='Jobs',
                  color_discrete_sequence=['#4f8ef7'],
                  text='Jobs')
    fig1.update_traces(textposition='outside', textfont=dict(color='#f0f4ff', size=11))
    fig1.update_layout(**make_layout({'coloraxis_showscale': False}))
    fig1.update_xaxes(tickangle=-30)

    # Chart 2 — Fixed vs Hourly donut
    rt = df['rate_type'].value_counts().reset_index()
    rt.columns = ['Type','Count']
    fig2 = px.pie(rt, names='Type', values='Count', hole=0.5,
                  color_discrete_sequence=['#4f8ef7','#22d3a0'])
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                       font=dict(color='#f0f4ff', family='Inter'),
                       margin=dict(t=10,b=10,l=10,r=10),
                       legend=dict(bgcolor='rgba(0,0,0,0)'))

    # Chart 3 — Avg price by country
    cp = df.groupby('client_country')['price_usd'].mean().reset_index()
    cp.columns = ['Country','Avg Price USD']
    cp = cp.sort_values('Avg Price USD', ascending=False).head(10)
    fig3 = px.bar(cp, x='Avg Price USD', y='Country', orientation='h',
                  color='Avg Price USD', color_continuous_scale='Teal')
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(color='#f0f4ff', family='Inter'),
                       showlegend=False, coloraxis_showscale=False,
                       margin=dict(t=10,b=20,l=130,r=20))
    fig3.update_yaxes(autorange='reversed')

    # Chart 4 — Rating vs Price scatter
    # FIX: Extreme price outliers (up to $37,800) pushed all dots near y=0.
    # Cap y-axis at 95th percentile so the bulk of data fills the chart area.
    sample = df.sample(min(800, len(df)), random_state=42)
    price_cap = float(df['price_usd'].quantile(0.95))
    fig4 = px.scatter(sample, x='client_average_rating', y='price_usd',
                      color='rate_type', opacity=0.75,
                      color_discrete_map={'fixed':'#4f8ef7','hourly':'#f5a623'},
                      labels={'client_average_rating':'Client Rating','price_usd':'Price (USD)'})
    fig4.update_traces(marker=dict(size=6))
    fig4.update_layout(**make_layout())
    fig4.update_yaxes(range=[0, price_cap])

    return render_template('index.html',
        total=total, avg_price=avg_price, avg_rating=avg_rating, countries=countries,
        chart1=fig_json(fig1), chart2=fig_json(fig2),
        chart3=fig_json(fig3), chart4=fig_json(fig4))

# ── STATISTICS ────────────────────────────────────────────────
@app.route('/stats')
def stats_page():
    desc = df[['price_usd','min_price','max_price','client_average_rating','client_review_count']].describe().round(2)
    desc.index = ['Count','Mean','Std','Min','25%','Median','75%','Max']
    desc_data = desc.reset_index().rename(columns={'index':'Statistic'}).to_dict('records')

    top8 = df['client_country'].value_counts().head(8).index.tolist()
    ci_rows = []
    for c in top8:
        sub = df[df['client_country']==c]['price_usd']
        n = len(sub); mean = sub.mean(); se = stats.sem(sub)
        ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
        ci_rows.append({'Country':c,'N':n,'Mean (USD)':round(mean,2),
                        'CI Lower':round(ci[0],2),'CI Upper':round(ci[1],2)})
    ci_df = pd.DataFrame(ci_rows)

    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(
        x=ci_df['Country'], y=ci_df['Mean (USD)'],
        error_y=dict(type='data',
                     array=(ci_df['CI Upper']-ci_df['Mean (USD)']).tolist(),
                     arrayminus=(ci_df['Mean (USD)']-ci_df['CI Lower']).tolist()),
        mode='markers+lines',
        marker=dict(color='#4f8ef7', size=10),
        line=dict(color='#4f8ef7', width=1.5, dash='dot')))
    fig_ci.update_layout(**make_layout())
    fig_ci.update_xaxes(tickangle=-20)

    earn = df['price_usd']
    # FIX: Clip to 95th percentile to remove extreme outliers.
    # Use histnorm='probability density' so histogram and normal PDF are on the same scale.
    earn_clip = earn[earn <= earn.quantile(0.95)]
    mu, sigma = stats.norm.fit(earn_clip)
    x_range = np.linspace(earn_clip.min(), earn_clip.max(), 300)
    # PDF values directly match 'probability density' histnorm — no manual scaling needed.
    pdf_vals = stats.norm.pdf(x_range, mu, sigma)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=earn_clip, nbinsx=50,
        histnorm='probability density',
        name='Actual Distribution',
        marker=dict(color='#4f8ef7', line=dict(color='rgba(255,255,255,0.15)', width=0.5)),
        opacity=0.80
    ))
    fig_dist.add_trace(go.Scatter(
        x=x_range, y=pdf_vals, mode='lines',
        name='Normal Fit',
        line=dict(color='#f5a623', width=2.5)
    ))
    fig_dist.update_layout(**make_layout({
        'xaxis_title': 'Price (USD) — clipped at 95th pct',
        'yaxis_title': 'Probability Density',
        'bargap': 0.05
    }))

    heat_df = df[df['client_country'].isin(top8)]
    heat_pivot = heat_df.groupby(['client_country','rate_type'])['price_usd'].mean().unstack().round(0)
    # FIX: 'Blues' is near-invisible on dark background (dark blue on dark bg = invisible).
    # Use 'Viridis' which has yellow/green highlights that pop on dark themes.
    # Add text annotations so exact values are readable regardless of color.
    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale='Viridis',
        aspect='auto',
        text_auto='.0f'
    )
    fig_heat.update_traces(textfont=dict(color='white', size=13))
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f0f4ff', family='Inter'),
        margin=dict(t=10, b=10, l=130, r=10),
        coloraxis_colorbar=dict(
            tickfont=dict(color='#f0f4ff'),
            titlefont=dict(color='#f0f4ff')
        )
    )

    # FIX: Removed y-axis range cap — it was hiding outlier dots above the cutoff.
    # 0.999 clip removes only extreme single outliers, keeps the visible outlier cluster.
    # Increased marker size+opacity and set outliercolor='#f5a623' so dots clearly show.
    box_cap = float(df['price_usd'].quantile(0.999))
    fig_box = px.box(
        df[df['price_usd'] <= box_cap],
        x='rate_type', y='price_usd',
        color='rate_type',
        points='outliers',
        color_discrete_map={'fixed': '#4f8ef7', 'hourly': '#22d3a0'}
    )
    fig_box.update_traces(
        marker=dict(size=5, opacity=0.7, outliercolor='#f5a623'),
        line=dict(width=2)
    )
    fig_box.update_layout(**make_layout({'showlegend': False}))

    return render_template('stats.html',
        desc_data=desc_data, ci_table=ci_rows,
        mu=round(mu,2), sigma=round(sigma,2),
        fig_ci=fig_json(fig_ci), fig_dist=fig_json(fig_dist),
        fig_heat=fig_json(fig_heat), fig_box=fig_json(fig_box))

# ── PREDICTOR ─────────────────────────────────────────────────
@app.route('/predict', methods=['GET','POST'])
def predict():
    countries_list = sorted(df['client_country'].unique().tolist())
    prediction = None
    form_data  = {}

    sel_min    = 100.0
    sel_max    = 200.0
    sel_rate   = 'fixed'
    sel_rating = 4.5
    sel_reviews= 10.0

    if request.method == 'POST':
        try:
            sel_rate    = request.form['rate_type']
            sel_rating  = float(request.form['rating'])
            sel_reviews = float(request.form['reviews'])
            sel_min     = float(request.form['min_price'])
            sel_max     = float(request.form['max_price'])
            prediction  = predict_price(sel_min, sel_max, sel_rate, sel_rating, sel_reviews)
            form_data   = request.form
        except Exception as e:
            prediction = f"Error: {e}"

    # Graph 1 — Scatter + Regression Line
    sample = df.sample(min(500, len(df)), random_state=1)
    fig_reg = px.scatter(sample, x='min_price', y='price_usd',
                         color='rate_type', opacity=0.4,
                         color_discrete_map={'fixed':'#4f8ef7','hourly':'#f5a623'},
                         labels={'min_price':'Min Price (USD)','price_usd':'Avg Price (USD)'})
    x_line = np.linspace(df['min_price'].quantile(0.01), df['min_price'].quantile(0.95), 100)
    y_line = [predict_price(xi, xi*2, sel_rate, sel_rating, sel_reviews) for xi in x_line]
    fig_reg.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                  name='Regression Line',
                                  line=dict(color='#22d3a0', width=2.5, dash='dash')))
    if prediction and isinstance(prediction, float):
        fig_reg.add_trace(go.Scatter(x=[sel_min], y=[prediction], mode='markers',
                                      name='Your Prediction',
                                      marker=dict(color='#f5a623', size=14, symbol='star')))
    fig_reg.update_layout(**make_layout())

    # Graph 2 — Rating vs Predicted Price
    rating_range = np.arange(1.0, 5.1, 0.1)
    pred_by_rating = [predict_price(sel_min, sel_max, sel_rate, r, sel_reviews) for r in rating_range]
    fig_rat = go.Figure()
    fig_rat.add_trace(go.Scatter(x=list(rating_range), y=pred_by_rating,
                                  mode='lines', name='Predicted Price',
                                  line=dict(color='#a78bfa', width=2.5),
                                  fill='tozeroy', fillcolor='rgba(167,139,250,0.08)'))
    if prediction and isinstance(prediction, float):
        fig_rat.add_trace(go.Scatter(x=[sel_rating], y=[prediction], mode='markers',
                                      name='Your Input',
                                      marker=dict(color='#f5a623', size=12, symbol='star')))
    fig_rat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='#f0f4ff', family='Inter'),
                           xaxis_title='Client Rating', yaxis_title='Predicted Price (USD)',
                           margin=dict(t=20,b=50,l=55,r=20),
                           legend=dict(bgcolor='rgba(0,0,0,0)'))

    # Graph 3 — Min Price vs Predicted Price
    min_range = np.linspace(10, 2000, 60)
    pred_by_min = [predict_price(m, m*2, sel_rate, sel_rating, sel_reviews) for m in min_range]
    fig_minp = go.Figure()
    fig_minp.add_trace(go.Scatter(x=list(min_range), y=pred_by_min,
                                   mode='lines', name='Predicted Price',
                                   line=dict(color='#22d3a0', width=2.5),
                                   fill='tozeroy', fillcolor='rgba(34,211,160,0.08)'))
    if prediction and isinstance(prediction, float):
        fig_minp.add_trace(go.Scatter(x=[sel_min], y=[prediction], mode='markers',
                                       name='Your Input',
                                       marker=dict(color='#f5a623', size=12, symbol='star')))
    fig_minp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f0f4ff', family='Inter'),
                            xaxis_title='Min Price (Input)', yaxis_title='Predicted Price (USD)',
                            margin=dict(t=20,b=50,l=55,r=20),
                            legend=dict(bgcolor='rgba(0,0,0,0)'))

    return render_template('predict.html',
        fig_reg=fig_json(fig_reg), fig_rat=fig_json(fig_rat), fig_minp=fig_json(fig_minp),
        r2=r2, countries_list=countries_list, prediction=prediction, form_data=form_data)

if __name__ == '__main__':
    import os
    # ── COLAB / CLOUD RUN SUPPORT ──────────────────────────────────────
    # In Google Colab, Flask's localhost is not accessible from your browser.
    # Two options to get a public URL:
    #
    # OPTION 1 — pyngrok (recommended, free):
    #   !pip install pyngrok -q
    #   from pyngrok import ngrok
    #   ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # free at ngrok.com
    #   public_url = ngrok.connect(5000)
    #   print("Public URL:", public_url)
    #   app.run()
    #
    # OPTION 2 — flask-ngrok (simplest):
    #   !pip install flask-ngrok -q
    #   from flask_ngrok import run_with_ngrok
    #   run_with_ngrok(app)
    #   app.run()
    #
    # OPTION 3 — Local run (normal PC / VS Code):
    #   Just run this file normally: python app.py
    #   Then open http://127.0.0.1:5000 in your browser.
    # ───────────────────────────────────────────────────────────────────
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
