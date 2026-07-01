import pandas as pd
import numpy as np
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from wordviz.loading import EmbeddingLoader
from wordviz.plotting import Visualizer
from transformers_interpret import SequenceClassificationExplainer

# files
DATA_DIR = Path('../data/processed')
MODEL_DIR = Path('../src/models/bert-reviews-tuned')

df = pd.read_csv(DATA_DIR / 'reviews_clean.csv')
test_df = pd.read_csv(DATA_DIR / 'test_results.csv')
emb_df = pd.read_csv(DATA_DIR / 'cls_embeddings.csv')
prod_total = pd.read_csv(DATA_DIR / 'prod_aggregated.csv')

MIN_REVIEWS = 5
COLOR_MAP = {'1':'#E24B4A','2':"#ffae42",'3':'#888780','4':'#1D9E75','5':'#0F6E56'}
SENTIMENT_COLOR_MAP = {'negative':'#E24B4A','neutral':'#888780','positive':'#1D9E75'}
INT_TO_NAME = {0:'negative', 1:'neutral', 2:'positive'}

# variables: product aggregation by year
prod_year = (
    df.groupby(['ProductId','year']).agg(
        n_reviews  = ('Score','size'),
        mean_score = ('Score','mean'),
        score_std  = ('Score','std'),
        pct_negative = ('label', lambda x: (x == 0).mean()),
    )
    .reset_index()
)
# keep only products with enough reviews across all years
valid_ids = prod_total['ProductId'].values
prod_year = prod_year[prod_year['ProductId'].isin(valid_ids)].copy()
prod_year['score_std'] = prod_year['score_std'].fillna(0)
years_available = sorted(prod_year['year'].unique())


emb_labels = emb_df['label'].tolist()
emb_matrix = emb_df.drop(columns='label').values

# with wordviz better to pre calculate figures
loader = EmbeddingLoader()
loader.load_contextual(embeddings=emb_matrix, labels=emb_labels, embedding_type='sentence')
viz = Visualizer(loader)

fig_pca  = viz.plot_interactive(red_method='pca',  theme='light1', title='Reviews BERT embeddings PCA')
fig_tsne = viz.plot_interactive(red_method='tsne', theme='light1', title='Reviews BERT embeddings T-SNE')

def get_emb_figure(method):
    return fig_pca if method == 'pca' else fig_tsne

# users
user_stats = df.groupby('UserId').agg(
    n_reviews   = ('Score','size'),
    mean_score  = ('Score','mean'),
    years_active = ('year', lambda x: x.max() - x.min()),
).reset_index()
threshold = user_stats['n_reviews'].quantile(0.95)
user_stats['segment'] = user_stats['n_reviews'].apply(
    lambda n: 'usual' if n >= threshold else 'occasional'
)


app = Dash(__name__, title='Amazon Fine Food Reviews Dashboard', suppress_callback_exceptions=True)

LABEL_STYLE = dict(fontSize=14, color='#5F5E5A', marginBottom=4)
TAB_STYLE = dict(padding='8px 20px', fontSize=14)
TAB_SEL = dict(padding='8px 20px', fontSize=14, fontWeight='500', borderTop='2px solid #7F4BC4', color='#7F4BC4')

app.layout = html.Div([
    html.Div([
        html.H2('Amazon Fine Food Reviews Dashboard', style=dict(margin='0 0 2px', fontSize=20, fontWeight=500)),
        html.P('NLP project about Data Analytics and Sentiment Analysis',
               style=dict(margin=0, fontSize=14, color='#888780')),
    ], style=dict(padding='20px 28px 12px')),

    dcc.Tabs(id='tabs', value='overview', children=[
        dcc.Tab(label='Overview', value='overview', style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label='Products', value='products', style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label='Embeddings', value='embeddings', style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label='Live inference', value='live', style=TAB_STYLE, selected_style=TAB_SEL),
    ]),

    html.Div(id='tab-content', style=dict(padding='24px 28px')),
], style=dict(fontFamily='sans-serif', maxWidth=1100, margin='0 auto'))


# tab 1: overview
def layout_overview():
    score_counts = df['Score'].value_counts().sort_index()

    # figA: score distribution
    figA = go.Figure()
    for score, count in score_counts.items():
        figA.add_trace(go.Bar(x=[score], y=[count], marker_color=COLOR_MAP[str(score)], name=str(score), showlegend=False,))
    figA.update_layout(title='Score distribution', xaxis_title='Score', yaxis_title='Number of reviews', margin=dict(t=40,b=40,l=40,r=10), height=320)

    # figB: reviews per year
    yearly = df.groupby('year').size().reset_index(name='count')
    figB = px.bar(yearly, x='year', y='count', title='Review volume by year', 
            labels={'count': 'Number of reviews', 'year': 'Year'}, color_discrete_sequence=['#B5D4F4'])
    figB.update_layout(margin=dict(t=40,b=40,l=40,r=10), height=320)

    # figC: length distribution
    figC = go.Figure(go.Histogram(x=df['review_len'], marker_color='#636EFA', name='Length'))
    figC.update_layout(title='Review length distribution', xaxis_title='Words',
                    yaxis_title='Number of reviews', margin=dict(t=40,b=40,l=40,r=10), height=320, showlegend=False)

    # figD: usual vs occasional users
    figD = go.Figure()

    for seg, color in [('usual', '#185FA5'), ('occasional', '#9FE1CB')]:
        sub = user_stats[user_stats['segment'] == seg]
        figD.add_trace(go.Histogram(x=sub['mean_score'], name=seg, opacity=0.7, marker_color=color, nbinsx=5, histnorm='probability density'))

    figD.update_layout(
        title='Score distribution by segment', xaxis_title='Mean score', yaxis_title='Density',
        height=320,  margin=dict(t=50, b=40, l=40, r=10),
        legend=dict(orientation='h', y=-0.2, x=0.0), barmode='overlay'
    )

    grid = dict(display='grid', gridTemplateColumns='1fr 1fr', gap='20px')

    return html.Div([
        html.Div([dcc.Graph(figure=figA), dcc.Graph(figure=figB)], style=grid),
        html.Div([dcc.Graph(figure=figC), dcc.Graph(figure=figD)], style=grid),
    ])


# tab 2: products
def layout_products():
    return html.Div([
        dcc.Graph(id='quadrant-graph'),

        html.Div([
            html.Span('Only show products with at least ', style=dict(fontSize=14, color='#5F5E5A')),
            dcc.Input(id='min-reviews-input', type='number', value=MIN_REVIEWS,
                min=1, step=1,
                style=dict(width=60, fontSize=14, marginLeft=6, marginRight=4)),
            html.Span('reviews in that year', style=dict(fontSize=14, color='#5F5E5A')),
        ], style=dict(marginBottom=20)),

        html.Div([
            html.Label('year', style=LABEL_STYLE),
            dcc.Slider(
                id='year-slider',
                min=int(years_available[0]), max=int(years_available[-1]),
                step=1,
                value=int(years_available[-1]),
                marks={int(y): str(y) for y in years_available if y % 2 == 0},
                tooltip=dict(placement='bottom', always_visible=True),
            ),
        ], style=dict(marginBottom=24)),
    ])


# tab 3: embeddings
def layout_embeddings():
    emb_controls = html.Div([
        html.Label('Reduction Method', style=LABEL_STYLE),
        dcc.Dropdown(
            id='emb-method',
            options=[{'label':'PCA','value':'pca'},{'label':'t-SNE','value':'tsne'}],
            value='pca', clearable=False, searchable=False,
            style=dict(width=160, fontSize=14),
        ),
    ], style=dict(marginBottom=16))

    return html.Div([
            html.Div([emb_controls, dcc.Graph(id='emb-graph')], style=dict(flex=1, minWidth=0)),
        ], style=dict(display='flex', gap=24, alignItems='flex-start')),


# tab 4: live inference
# test review to paste: This product is terribile, It seemed to be good from the image online, but when we ate it, it was absolutely disgusting
def layout_live():
    return html.Div([
        html.P(
            'Enter a review in English. The BERT model will classify it and return the sentiment class and confidence. Processing time on CPU may take some seconds.',
            style=dict(fontSize=14, color='#5F5E5A', marginBottom=16),
        ),
        dcc.Textarea(
            id='review-input',
            placeholder='Type or paste a review…',
            style=dict(width='100%', height=120, fontSize=14, padding=10, borderRadius=8, border='0.5px solid #B4B2A9',
                fontFamily='sans-serif', boxSizing='border-box'),
        ),
        html.Div([
            html.Button('Analyze', id='analyze-btn', n_clicks=0,
                style=dict(marginTop=12, padding='8px 20px', fontSize=14,
                cursor='pointer', borderRadius=8, border='0.5px solid #B4B2A9', background='white')),
            html.Button('Clear', id='clear-btn', n_clicks=0,
                style=dict(marginTop=12, marginLeft=8, padding='8px 20px',
                fontSize=14, cursor='pointer', borderRadius=8, border='0.5px solid #B4B2A9', background='white')),
        ]),
        dcc.Loading(
            type='dot', delay_show=100,
            children=html.Div(id='inference-output', style=dict(marginTop=24)),
        ),
    ])


# callbacks

@app.callback(Output('tab-content','children'), Input('tabs','value'))
def render_tab(tab):
    if tab == 'overview': return layout_overview()
    if tab == 'products': return layout_products()
    if tab == 'embeddings': return layout_embeddings()
    if tab == 'live': return layout_live()

@app.callback(
    Output('quadrant-graph','figure'),
    Input('year-slider','value'),
    Input('min-reviews-input','value'),
)
def update_quadrant(year, min_rev):
    min_rev = min_rev or 1
    sub = prod_year[(prod_year['year'] == year) & (prod_year['n_reviews'] >= min_rev)].copy()

    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(text='No data available', showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5, font=dict(size=14))
        fig.update_layout(height=480, margin=dict(t=40,b=40,l=40,r=10))
        return fig

    fig = px.scatter(
        sub, x='mean_score', y='score_std', size='n_reviews', color='pct_negative',
        color_continuous_scale='RdYlGn_r',
        hover_data={'ProductId':True,'n_reviews':True, 'mean_score':':.2f','score_std':':.2f','pct_negative':':.1%'},
        opacity=0.7,
        labels={'mean_score':'Mean score','score_std':'Score std dev', 'pct_negative':'% negative','n_reviews':'Reviews'},
        title=f'Product quadrant: {year}',
    )
    fig.update_layout(height=480, margin=dict(t=50,b=40,l=50,r=10))
    return fig


@app.callback(Output('emb-graph','figure'), Input('emb-method','value'))
def update_embeddings(method):
    return get_emb_figure(method)

@app.callback(Output('review-input','value'), Input('clear-btn','n_clicks'), prevent_initial_call=True)
def clear_input(_):
    return ''

@app.callback(
    Output('inference-output','children'),
    Input('analyze-btn','n_clicks'),
    State('review-input','value'),
    prevent_initial_call=True,
)
def run_inference(n_clicks, text):
    if not text or not text.strip():
        return html.P('Please enter some text', style=dict(color='#888780', fontSize=14))

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch.nn.functional as F
    except ImportError:
        return html.P('libraries not available', style=dict(color='#E24B4A', fontSize=14))

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    inputs = tokenizer(text.strip(), return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    pred  = int(torch.argmax(probs))
    label = INT_TO_NAME[pred]
    color = SENTIMENT_COLOR_MAP[label]

    confidence_lines = [
        html.P(f'{INT_TO_NAME[i]}: {float(probs[i]):.1%}',
               style=dict(fontSize=14, color=SENTIMENT_COLOR_MAP[INT_TO_NAME[i]], margin='2px 0'))
        for i in range(3)
    ]

    # explainability
    xai_output = []
    try:
        text_trunc = text.strip()
        tokens = tokenizer.tokenize(text_trunc)
        if len(tokens) > 510:
            text_trunc = tokenizer.convert_tokens_to_string(tokens[:510])

        cls_explainer  = SequenceClassificationExplainer(model, tokenizer)
        font_override = '<style>* { font-family: sans-serif !important; }</style>'
        result = cls_explainer(text_trunc) 
        html_viz   = font_override + cls_explainer.visualize()._repr_html_()

        xai_output = [
            html.Hr(style=dict(margin='20px 0', borderColor='#D3D1C7')),
            html.P('Explainability (Integrated Gradients)', style=dict(fontSize=14, fontWeight=500, marginBottom=8)),
            html.Iframe(srcDoc=html_viz, style=dict(width='100%', height=180, border='none', borderRadius=8, background='white'),),
        ]
    except Exception:
        pass   # if transformers-interpret not available, skip explanation

    return html.Div([
        html.Div([
            html.Span('Prediction: ', style=dict(fontSize=14)),
            html.Span(label.upper(), style=dict(fontSize=18, fontWeight=500, color=color)),
        ], style=dict(marginBottom=16)),
        html.Div(confidence_lines),
        *xai_output,
    ], style=dict(background='#F1EFE8', borderRadius=12, padding='16px 20px', fontFamily='sans-serif'))


if __name__ == '__main__':
    app.run(debug=True)