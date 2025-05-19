import streamlit as st
import pandas as pd
import nltk
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.warning("Downloading missing NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)

ensure_nltk_data()

import plotly.express as px
import plotly.graph_objects as go
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.util import ngrams
from whoosh import index, fields, qparser
import watchdog

SENTIMENT_COLORS = {
    'positive': '#2ca02c',   
    'neutral':  '#7f7f7f',   
    'negative': '#1f77b4'    
}

# Whoosh setup
IDX_DIR = "whoosh_index"
schema = fields.Schema(idx=fields.NUMERIC(stored=True, unique=True),
                       text=fields.TEXT(stored=True))

if not os.path.exists(IDX_DIR):
    os.mkdir(IDX_DIR)
    ix = index.create_in(IDX_DIR, schema)
else:
    try:
        ix = index.open_dir(IDX_DIR)
    except Exception as e:
        st.warning(f"Index corrupted: {e}. Rebuilding index...")
        os.rename(IDX_DIR, f"{IDX_DIR}_backup_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}")
        os.mkdir(IDX_DIR)
        ix = index.create_in(IDX_DIR, schema)

def whoosh_search(query:str, top_k:int|None=None):
    """
    Returns every hit (unless top_k is an int).
    """
    if not query.strip():
        return []

    try:
        with ix.searcher() as searcher:
            parser = qparser.MultifieldParser(["text"], schema=ix.schema)
            q = parser.parse(query)

            # None → no limit
            results = searcher.search(q, limit=top_k)

            hits = [{
                "idx": hit["idx"],
                "score": round(hit.score, 2),
                "snippet": hit["text"]
            } for hit in results]

        return hits

    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# Set page configuration
st.set_page_config(
    page_title="V2G/V2H Reddit Analysis - ePowerMove",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

TOPIC_INFO = {
    "Energy Resilience and V2G/V2H Backup": """
Electric vehicles can feed **homes** (V2H) or the **grid** (V2G),
providing backup power and grid-stabilisation in addition to transport.
Users compare this to stationary batteries (e.g. Powerwall) and gas
generators, weighing cost, convenience, and reliability.
""",

    "Remuneration and User Control": """
Focus on the financial upside of bidirectional charging: selling cheap
off-peak energy back at peak prices, earning credits via **net-metering**,
and securing perks such as free chargers or parking—provided the user can
set price thresholds and battery-reserve limits.
""",

    "Hardware and Wiring": """
What it really takes to interface an EV with home circuits or the grid:
transfer switches, inverters, breakers, distribution panels, and how cost
and complexity differ between V2H, V2G, and V2L setups.
""",

    "EV Bi-directional Market Landscape": """
Brand-by-brand look at bidirectional capabilities: Tesla (Powershare),
Ford F-150 Lightning, Rivian, Hyundai/Kia’s E-GMP platform, and more—along
with pricing, availability, and service considerations.
""",

    "Bi-directional Charging Standards": """
How CCS, NACS, CHAdeMO and protocol layers (ISO 15118, IEEE 1547, etc.)
shape the rollout of V2G/V2H worldwide, plus the regional politics behind
each standard.
""",

    "V2X Preferences and Trade-offs": """
Users compare **V2L, V2H, and V2G**: V2L’s plug-and-play convenience for
appliances or camping, V2H’s whole-home backup during outages, and V2G’s
potential grid revenue—balanced against cost, complexity, and battery wear.
""",

    "Battery Longevity": """
Explores how charge–discharge cycling, chemistry choice (e.g. LFP), and
emerging “million-mile” cells affect battery warranties and long-term
capacity in bidirectional scenarios.
""",

    "Solar Integration and Inverter Ecosystems": """
Covers the solar-plus-EV ecosystem: Enphase micro-inverters, SolarEdge and
Sol-Ark string inverters, all-in-one controllers, and how they orchestrate
PV, batteries, EVs, and the grid.
""",

    "Electric School Buses and V2G": """
Fleet electrification—especially school buses—offers predictable idle
windows that make them ideal mobile battery banks for peak-shaving and
disaster resilience.
"""
}

EMOJI = {          # 1-to-1 mapping topic → emoji
    "Energy Resilience and V2G/V2H Backup": "🏠",
    "Remuneration and User Control": "💰",
    "Hardware and Wiring": "🛠️",
    "EV Bi-directional Market Landscape": "🚗",
    "Bi-directional Charging Standards": "🔌",
    "V2X Preferences and Trade-offs": "⚡",
    "Battery Longevity": "🔋",          # reuse if you like
    "Solar Integration and Inverter Ecosystems": "☀️",
    "Electric School Buses and V2G": "🚌"
}

# --- V2X analysis helpers ---
def analyze_v2x_mentions(df_comments):
    """
    Build DataFrame of Topic × Sentiment × Mode (V2H/V2G/V2L) counts and percentages.
    """
    patterns = {
        'V2H': re.compile(
            r'\b('
            r'v2h|'               # standard acronym
            r'vth|'               # common mistype
            r'vehicle[- ]to[- ]home|'   # full phrase 1
            r'vehicle[- ]to[- ]house'   
            r')\b',
            re.IGNORECASE
        ),
        'V2G': re.compile(r'\b(v2g|vtg|vehicle[- ]to[- ]grid)\b',  re.IGNORECASE),
        'V2L': re.compile(r'\b(v2l|vtl|vehicle[- ]to[- ]load)\b',  re.IGNORECASE)
    }

    required_columns = ['Comment', 'Topic_Label', 'Sentiment']
    if not all(col in df_comments.columns for col in required_columns) or df_comments.empty:
        st.warning("Invalid or empty data for V2X analysis.")
        return pd.DataFrame()

    matches = {mode: df_comments['Comment'].apply(lambda x: bool(p.search(str(x))))
               for mode, p in patterns.items()}

    rows = []
    topics = df_comments['Topic_Label'].unique()
    sentiments = ['positive', 'neutral', 'negative']
    for topic in topics:
        mask_topic = df_comments['Topic_Label'] == topic
        for sentiment in sentiments:
            mask_sent = df_comments['Sentiment'] == sentiment
            mask = mask_topic & mask_sent
            total = mask.sum()
            for mode in patterns.keys():
                cnt = matches[mode][mask].sum()
                pct = cnt / total * 100 if total else 0
                rows.append({
                    'Topic': topic,
                    'Sentiment': sentiment.title(),
                    'Mode': mode,
                    'Count': cnt,
                    'Percentage': pct
                })
    return pd.DataFrame(rows)

def visualize_v2x_data(df):
    required_columns = ['Topic', 'Sentiment', 'Mode', 'Count', 'Percentage']
    if df.empty or not all(col in df.columns for col in required_columns):
        return
    with st.expander("ℹ️ What counts as a V2X mention?"):
        st.markdown("""
        This analysis only includes **explicit keyword matches** for V2H, V2G, or V2L—
        such as *"vehicle-to-grid"*, *"V2L"*, or *"V2H"*.
        
        Comments that **describe** bidirectional use cases (e.g. powering a home,
        selling energy to the grid) but **don’t use these terms directly**
        may not appear in this breakdown. Hence, interpret the results with caution.
        """, unsafe_allow_html=True)

    st.subheader("V2H/V2G/V2L Mention Rates by Topic & Sentiment")
    df_disp = df.copy()
    df_disp['Percentage'] = df_disp['Percentage'].round(1).astype(str) + '%'
    st.dataframe(df_disp, use_container_width=True)

    capitalized_sentiment_colors = {
        'Positive': SENTIMENT_COLORS['positive'],  
        'Neutral': SENTIMENT_COLORS['neutral'],    
        'Negative': SENTIMENT_COLORS['negative']   
    }
    fig = px.bar(
        df,
        x='Mode', y='Percentage', color='Sentiment',
        facet_col='Topic', facet_col_wrap=3,
        text=df['Percentage'].round(1).astype(str) + '%',
        color_discrete_map=capitalized_sentiment_colors,
        category_orders={'Mode': ['V2H', 'V2G', 'V2L'],
                        'Sentiment': ['Positive', 'Neutral', 'Negative']}
    )
    fig.update_layout(
        xaxis_title="Mode",
        yaxis_title="% of comments",
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=50)
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .block-container {
        padding: 1rem 2rem;
    }
    h1 {
        color: #1E3A8A;
        margin-bottom: 1.5rem;
    }
    h2 {
        color: #2563EB;
        padding-top: 1rem;
        margin-bottom: 1rem;
    }
    h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .comment-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metrics-container {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .highlight-card {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .positive-card {
        border-left: 5px solid #2ca02c;
        background-color: #e9f5e9;
    }
    .negative-card {
        border-left: 5px solid #1f77b4;
        background-color: #e5f0fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 0px;
        padding-right: 0px;
    }
    div[data-testid="stSidebar"] > div:first-child {
        background-color: #F1F5F9;
    }
    div[data-testid="stSidebar"] li {
        margin-bottom: 0.75rem;
    }
    div.stRadio > div {
        padding: 10px;
        background-color: #F9FAFB;
        border-radius: 0.5rem;
    }
    div.stSlider > div {
        padding: 10px 0;
    }
    div.stMetric {
        background-color: #F3F4F6;
        padding: 10px;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .topic-description {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-left: 3px solid #6B7280;
    }
    .pagination-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        margin: 20px 0;
    }
    .pagination-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1rem;
        background-color: #EFF6FF;
        color: #2563EB;
        border-radius: 0.375rem;
        font-weight: 500;
        cursor: pointer;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: all 0.15s ease;
    }
    .pagination-button:hover {
        background-color: #DBEAFE;
    }
    .pagination-info {
        color: #4B5563;
        font-weight: 500;
    }
    .stSelectbox > div > div {
        background-color: #F9FAFB;
    }
    .search-container {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }
    /* Custom sentiment badges */
    .sentiment-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: capitalize;
    }
    .sentiment-positive {
        background-color: rgba(44, 160, 44, 0.1);  
        color: #2ca02c;
    }
    .sentiment-neutral {
       background-color: rgba(127, 127, 127, 0.1); 
       color: #7f7f7f;
    }
    .sentiment-negative {
       background-color: rgba(31, 119, 180, 0.1); 
       color: #1f77b4;
    }
    .highlight-keyword {
        background-color:#FFF7B2;   /* pale yellow */
        font-weight:700;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the data"""
    comments_path = "bertopic_comments_with_sentiment_updated.xlsx"
    summary_path = "bertopic_sentiment_analysis.xlsx"
    
    if not (os.path.exists(comments_path) and os.path.exists(summary_path)):
        st.error("Data files not found. Please check the file paths.")
        return None, None
    
    try:
        df_comments = pd.read_excel(comments_path)
        sentiment_summary = pd.read_excel(summary_path)
        return df_comments, sentiment_summary
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def extract_keywords(text_series, n_keywords=30):
    """Extract keywords from a series of text comments"""
    
    all_text = " ".join(text_series.tolist())
    stop_words = set(stopwords.words('english'))
    custom_stops = {'would', 'could', 'get', 'also', 'even', 'like', 'one', 'make', 'just', 'v2g', 'ev', 'vehicle', 'battery', 'power', 'grid', 'energy', 'time', 'use', 'using', 'used', 'car', 'thing'}
    stop_words.update(custom_stops)
    
    words = word_tokenize(all_text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
    
    word_freq = Counter(words)
    return word_freq.most_common(n_keywords)

@st.cache_data
def extract_bigrams(text_series, n_bigrams=30):
    """Extract bigrams from a series of text comments"""
    
    all_text = " ".join(text_series.tolist())
    stop_words = set(stopwords.words('english'))
    custom_stops = {'would', 'could', 'get', 'also', 'even', 'like', 'one', 'make', 'just', 'v2g', 'ev', 'battery', 'power', 'grid', 'time', 'use', 'using', 'used', 'car', 'thing'}
    stop_words.update(custom_stops)
    
    words = word_tokenize(all_text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
    
    bi_grams = ngrams(words, 2)
    bigram_freq = Counter([f"{w1} {w2}" for w1, w2 in bi_grams])
    return bigram_freq.most_common(n_bigrams)

@st.cache_data
def create_wordcloud(text_series):
    """Create a wordcloud from a series of text"""
    if text_series.empty:
        return None
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.warning("NLTK stopwords missing. Downloading...")
        nltk.download('stopwords')
    
    all_text = " ".join(text_series.tolist())
    stop_words = set(stopwords.words('english'))
    custom_stops = {'would', 'could', 'get', 'need', 'much', 'going', 'also', 'much', 'think', 'sure', 'even', 'like', 'one', 'every', 'somethings', 'bidirectional', 'said', 'see', 'going', 'evse', 'way', 'already', 'really', 'make', 'just', 'v2g', 'ev'}
    stop_words.update(custom_stops)
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        stopwords=stop_words,
        max_words=100,
        colormap='viridis'
    ).generate(all_text)
    
    return wordcloud

def get_sentiment_extremes(sentiment_summary):
    """Get the most positive and most negative topics"""
    most_positive = sentiment_summary.sort_values('Positive (%)', ascending=False).iloc[0]
    most_negative = sentiment_summary.sort_values('Negative (%)', ascending=False).iloc[0]
    return most_positive, most_negative

def format_sentiment_badge(sentiment):
    """Create HTML for a sentiment badge"""
    return f"""<span class="sentiment-badge sentiment-{sentiment}">{sentiment}</span>"""

def highlight_keywords(text: str, raw_query: str) -> str:
    """
    Wrap every term that appears in `raw_query`
    with <span class="highlight-keyword">…</span>.
    """
    if not raw_query.strip():
        return text

    # split on whitespace, ignore tiny words and boolean operators
    terms = [re.escape(t) for t in re.split(r'\s+', raw_query)
             if len(t) > 1 and t.lower() not in {"and", "or", "not"}]
    if not terms:
        return text

    pattern = re.compile(r'(' + '|'.join(terms) + r')', flags=re.I)
    return pattern.sub(r'<span class="highlight-keyword">\1</span>', text)

def custom_pagination(total_items, items_per_page, current_page, key_prefix):
    """Create a custom pagination component"""
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        if current_page > 1:
            if st.button("← Previous", key=f"{key_prefix}_prev"):
                st.session_state[f"{key_prefix}_page"] = current_page - 1
                st.rerun()
    
    with col2:
        st.markdown(f"""
        <div class="pagination-info">
            Page {current_page} of {total_pages} ({total_items} items)
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if current_page < total_pages:
            if st.button("Next →", key=f"{key_prefix}_next"):
                st.session_state[f"{key_prefix}_page"] = current_page + 1
                st.rerun()
    
    return current_page

# Main application
def main():
    # Load data
    df_comments, sentiment_summary = load_data()
    if df_comments is None or sentiment_summary is None:
        st.error("Failed to load data. Please check your file paths.")
        return
        
    if 'favorite_comments' not in st.session_state:
        st.session_state['favorite_comments'] = set()
    
    # Index comments only if necessary
    with st.spinner("Indexing comments..."):
        with ix.searcher() as searcher:
            indexed_ids = set(searcher.document_numbers())
        new_comments = df_comments.index[~df_comments.index.isin(indexed_ids)]
        
        if len(new_comments) > 0:
            writer = ix.writer()
            for idx in new_comments:
                comment = str(df_comments.loc[idx, "Comment"])
                writer.add_document(idx=idx, text=comment)
            writer.commit()
            st.write(f"🔍 Indexed {len(new_comments)} new comments.")
    
    # App title with logo
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <h1 style="margin-bottom: 0;">🔋 V2G/V2H Reddit Analysis Dashboard</h1>
    <a
      href="https://www.epowermove.eu/"
      target="_blank"
      style="margin-left: auto; margin-right: 1rem;
             padding: 0.25rem 0.75rem;
             background-color: #EFF6FF;
             border-radius: 9999px;
             font-weight: 600;
             color: #2563EB;
             text-decoration: none;"
    >
        ePowerMove
    </a>
</div>
""", unsafe_allow_html=True)
    
    # Initialize session state for page tracking
    if 'topic_page' not in st.session_state:
        st.session_state['topic_page'] = 1
    if 'explorer_page' not in st.session_state:
        st.session_state['explorer_page'] = 1
    
    # Sidebar for navigation with icons
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 1rem; border-bottom: 1px solid #E5E7EB;">
        <h2 style="margin-bottom: 0.5rem; color: #1E3A8A;">Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select page",
        ["🔍 Overview", "📊 Topic Analysis", "🔎 Comment Explorer", "⭐ Favourites", "📖 About"],
        format_func=lambda x: x.split(' ', 1)[1] if ' ' in x else x
    )
    
    # Overview page
    if page == "🔍 Overview":
        st.header("Overview of Sentiment Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Comments", len(df_comments))
        with col2:
            st.metric("Topics", df_comments['Topic_Label'].nunique())
        with col3:
            positive_pct = (df_comments['Sentiment'].value_counts(normalize=True).get('positive', 0) * 100).round(1)
            st.metric("Positive Sentiment", f"{positive_pct}%")
        
        most_positive, most_negative = get_sentiment_extremes(sentiment_summary)
        
        st.subheader("Sentiment Extremes by Topic")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='highlight-card positive-card'>
                <h3>Most Positive Topic</h3>
                <p><strong>{most_positive['Topic']}</strong></p>
                <p>Positive: {most_positive['Positive (%)']:.1f}%</p>
                <p>Neutral: {most_positive['Neutral (%)']:.1f}%</p>
                <p>Negative: {most_positive['Negative (%)']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class='highlight-card negative-card'>
                <h3>Most Negative Topic</h3>
                <p><strong>{most_negative['Topic']}</strong></p>
                <p>Positive: {most_negative['Positive (%)']:.1f}%</p>
                <p>Neutral: {most_negative['Neutral (%)']:.1f}%</p>
                <p>Negative: {most_negative['Negative (%)']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        overall_sentiment = (
            df_comments['Sentiment']
            .value_counts(normalize=True)
            .rename_axis('Sentiment')
            .reset_index(name='Percentage')
        )
        overall_sentiment['Percentage'] *= 100

        donut = px.pie(
            overall_sentiment,
            values='Percentage',
            names='Sentiment',
            color='Sentiment',
            hole=0.55,
            color_discrete_map=SENTIMENT_COLORS
        )
        donut.update_traces(texttemplate='%{percent:.1%}', textposition='inside')
        donut.update_layout(
            margin=dict(t=20, b=20, l=0, r=0),
            showlegend=True,
            annotations=[dict(
                text=f"<b>{len(df_comments)}</b><br>comments",
                showarrow=False,
                font_size=14
            )]
        )
                
        sentiment_summary_sorted = (
            sentiment_summary
            .sort_values('Negative (%)', ascending=False)
        )

        fig = go.Figure()

        colors = {
            'Positive (%)': SENTIMENT_COLORS['positive'],
            'Neutral (%)': SENTIMENT_COLORS['neutral'],
            'Negative (%)': SENTIMENT_COLORS['negative'],
        }
        sentiment_cols = ['Positive (%)', 'Neutral (%)', 'Negative (%)']

        # 2️⃣ build stacked bars
        for col in sentiment_cols:
            label = col.replace(' (%)', '')
            fig.add_trace(
                go.Bar(
                    y=sentiment_summary_sorted['Topic'],
                    x=sentiment_summary_sorted[col],
                    name=label,
                    orientation='h',
                    marker_color=colors[col],
                    text=sentiment_summary_sorted[col].round(1).astype(str) + '%',
                    textposition='auto'          # cleaner labels
                )
            )

        # 3️⃣ layout tweaks
        fig.update_layout(
            barmode='stack',
            yaxis={'categoryorder': 'total ascending'},
            legend_title_text='Sentiment',
            xaxis_title="Percentage (%)",
            yaxis_title="",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            margin=dict(l=10, r=10, t=10, b=10)
        )

        st.subheader("Sentiment Overview")          # one caption for the pair

        col1, col2 = st.columns([1, 2])             # 1/3 vs 2/3 width

        with col1:
            st.markdown("##### Overall")
            st.plotly_chart(donut, use_container_width=True)

        with col2:
            st.markdown("##### By Topic")
            st.plotly_chart(fig, use_container_width=True)

    # ── Topic Size Distribution ──
        topic_counts = (
            df_comments['Topic_Label']
              .value_counts()            # Series: index=topic_label, values=counts
              .rename_axis('Topic')      # name the index “Topic”
              .reset_index(name='Count') # turn it into DF with columns [Topic, Count]
        )
        
        # (2) Now Plotly will see exactly those two columns:
        fig_topics = px.bar(
            topic_counts,
            x='Topic',
            y='Count',
            text='Count',
            color='Count',
            color_continuous_scale='Viridis',
        )
        fig_topics.update_traces(textposition='outside')
        fig_topics.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Topic",
            yaxis_title="Number of Comments",
            margin=dict(t=150, b=100),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.subheader("Topic Size Distribution")
        st.plotly_chart(fig_topics, use_container_width=True)

        st.subheader("Topic Descriptions")

        col_left, col_right = st.columns(2)

        for i, (title, desc) in enumerate(TOPIC_INFO.items()):
            target_col = col_left if i % 2 == 0 else col_right
            icon = EMOJI.get(title, "•")  # fallback bullet if no emoji
            with target_col.expander(f"{icon}  {title}"):
                st.markdown(desc.strip())
                
    
    # Topic Analysis page
    elif page == "📊 Topic Analysis":
        st.header("Topic-Level Sentiment Analysis")
        
        st.markdown("""
        <div style="margin-bottom: 1rem; color: #4B5563; font-size: 0.9rem;">
            Select a topic below to analyse sentiment patterns and key discussion points within that topic.
        </div>
        """, unsafe_allow_html=True)
        
        selected_topic = st.selectbox(
            "Select Topic to Analyse", 
            options=df_comments['Topic_Label'].unique(),
            index=1
        )
        
        topic_comments = df_comments[df_comments['Topic_Label'] == selected_topic]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Comments in Topic", len(topic_comments))
        with col2:
            positive_pct = (topic_comments['Sentiment'].value_counts(normalize=True).get('positive', 0) * 100).round(1)
            st.metric("Positive Sentiment", f"{positive_pct}%")
        with col3:
            negative_pct = (topic_comments['Sentiment'].value_counts(normalize=True).get('negative', 0) * 100).round(1)
            st.metric("Negative Sentiment", f"{negative_pct}%")
        
        topic_sentiment = topic_comments['Sentiment'].value_counts(normalize=True).reset_index()
        topic_sentiment.columns = ['Sentiment', 'Percentage']
        topic_sentiment['Percentage'] = topic_sentiment['Percentage'] * 100
        
        st.subheader(f"Sentiment Distribution for: {selected_topic}")
        fig = px.bar(
            topic_sentiment,
            x='Sentiment',
            y='Percentage',
            color='Sentiment',
            color_discrete_map=SENTIMENT_COLORS,
            text=topic_sentiment['Percentage'].round(1).astype(str) + '%',
        )
        fig.update_layout(
            xaxis_title="", 
            yaxis_title="Percentage (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        sentiment_tab, keyword_tab, v2x_tab = st.tabs([
            "💬 Sentiment Comments", 
            "🔑 Keyword Analysis",
            "📊 V2X Mentions Breakdown"
        ])
        
        with sentiment_tab:
            st.markdown("""
            <div style="margin-bottom: 1rem; margin-top: 1rem;">
                <p style="margin-bottom: 0.5rem; font-weight: 500;">Filter comments by sentiment:</p>
            </div>
            """, unsafe_allow_html=True)
            
            selected_sentiment = st.radio(
                "Select Sentiment to Explore", 
                ["positive", "neutral", "negative"],
                horizontal=True,
                key="sentiment_filter"
            )
            
            filtered_comments = topic_comments[topic_comments['Sentiment'] == selected_sentiment]
            
            if len(filtered_comments) == 0:
                st.info(f"No {selected_sentiment} comments found for this topic.")
            else:
                st.subheader(f"{selected_sentiment.capitalize()} Comments ({len(filtered_comments)})")
                
                comments_per_page = 5
                
                if 'topic_sentiment_page' not in st.session_state:
                    st.session_state['topic_sentiment_page'] = 1
                
                if 'last_sentiment' not in st.session_state or st.session_state['last_sentiment'] != selected_sentiment:
                    st.session_state['topic_sentiment_page'] = 1
                    st.session_state['last_sentiment'] = selected_sentiment
                
                current_page = custom_pagination(
                    len(filtered_comments),
                    comments_per_page,
                    st.session_state['topic_sentiment_page'],
                    "topic_sentiment"
                )
                
                start_idx = (current_page - 1) * comments_per_page
                end_idx = min(start_idx + comments_per_page, len(filtered_comments))
                
                for i, (_, row) in enumerate(filtered_comments.iloc[start_idx:end_idx].iterrows(), start=start_idx+1):
                    st.markdown(f"""
                    <div class='comment-box'>
                        <strong>Comment {i}</strong>
                        <div style="margin-top: 0.5rem;">{row['Comment']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with keyword_tab:
            st.subheader("Keyword Analysis")
            
            st.markdown("""
            <div style="margin-bottom: 1rem; margin-top: 1rem;">
                <p style="margin-bottom: 0.5rem; font-weight: 500;">Filter keywords by sentiment:</p>
            </div>
            """, unsafe_allow_html=True)
            
            keyword_sentiment = st.radio(
                "Select Sentiment for Keyword Analysis", 
                ["All", "positive", "neutral", "negative"],
                horizontal=True,
                key="keyword_sentiment"
            )
            
            if keyword_sentiment == "All":
                keyword_comments = topic_comments['Comment']
            else:
                keyword_comments = topic_comments[topic_comments['Sentiment'] == keyword_sentiment]['Comment']
            
            if len(keyword_comments) > 0:
                keywords = extract_keywords(keyword_comments)
                bigrams = extract_bigrams(keyword_comments)
                
                word_tab, bigram_tab, cloud_tab = st.tabs(["Top Keywords", "Top Bigrams", "Word Cloud"])
                
                with word_tab:
                    st.subheader("Top Keywords")
                    keyword_df = pd.DataFrame(keywords, columns=['Keyword', 'Count'])
                    fig = px.bar(
                        keyword_df.head(15), 
                        x='Count', 
                        y='Keyword',
                        orientation='h',
                        color='Count',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with bigram_tab:
                    st.subheader("Top Bigrams")
                    bigram_df = pd.DataFrame(bigrams, columns=['Bigram', 'Count'])
                    fig = px.bar(
                        bigram_df.head(15), 
                        x='Count', 
                        y='Bigram',
                        orientation='h',
                        color='Count',
                        color_continuous_scale='Plasma'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with cloud_tab:
                    wordcloud = create_wordcloud(keyword_comments)
                    if wordcloud:
                        st.subheader("Word Cloud")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        plt.tight_layout()
                        st.pyplot(fig)
            else:
                st.info("Not enough comments for keyword analysis.")
        
        with v2x_tab:
            st.markdown("### Breakdown of V2H/V2G/V2L Mentions by Mode and Sentiment")

            modes = st.multiselect(
                "Select bidirectional modes to include",
                options=["V2H", "V2G", "V2L"],
                default=["V2H", "V2G", "V2L"],
                key="v2x_modes"
            )

            sents = st.multiselect(
                "Select sentiments to include",
                options=["Positive", "Neutral", "Negative"],
                default=["Positive", "Neutral", "Negative"],
                key="v2x_sents"
            )

            breakdown_df = analyze_v2x_mentions(topic_comments)  # scoped to current topic
            filtered_df = breakdown_df[
                breakdown_df["Mode"].isin(modes) &
                breakdown_df["Sentiment"].isin(sents)
            ]

            visualize_v2x_data(filtered_df)

    
    # Comment Explorer page
    elif page == "🔎 Comment Explorer":
        st.header("Comment Explorer")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_topic = st.selectbox(
                "Select Topic", 
                options=["All"] + list(df_comments['Topic_Label'].unique())
            )
        
        with col2:
            selected_sentiment = st.selectbox(
                "Select Sentiment", 
                options=["All", "positive", "neutral", "negative"]
            )
        
        search_term = st.text_input(
            "Search for keywords in comments:",
            placeholder="e.g.  v2g   battery   capacity",
            help="You can enter **multiple keywords** separated by spaces. "
        )
        
        if selected_topic == "All" and selected_sentiment == "All":
            filtered_comments = df_comments
        elif selected_topic == "All":
            filtered_comments = df_comments[df_comments['Sentiment'] == selected_sentiment]
        elif selected_sentiment == "All":
            filtered_comments = df_comments[df_comments['Topic_Label'] == selected_topic]
        else:
            filtered_comments = df_comments[(df_comments['Topic_Label'] == selected_topic) & 
                                           (df_comments['Sentiment'] == selected_sentiment)]
        
        snippets = {}
        if search_term:
            hits     = whoosh_search(search_term)
            idxs     = [h["idx"] for h in hits]          # every row that matched the search
            snippets = {h["idx"]: h["snippet"] for h in hits}

            # ⬇️ keep only rows that satisfy BOTH the dropdown filters AND the search hits
            filtered_comments = filtered_comments[filtered_comments.index.isin(idxs)]
        
        if 'last_explorer_filters' not in st.session_state:
            st.session_state['last_explorer_filters'] = (selected_topic, selected_sentiment, search_term)
        elif st.session_state['last_explorer_filters'] != (selected_topic, selected_sentiment, search_term):
            st.session_state['explorer_page'] = 1
            st.session_state['last_explorer_filters'] = (selected_topic, selected_sentiment, search_term)
        
        st.markdown(f"""
        <div style="margin: 1rem 0; padding: 0.75rem; background-color: #F3F4F6; border-radius: 0.5rem; text-align: center;">
            <span style="font-weight: 500;">Found {len(filtered_comments)} matching comments</span>
        </div>
        """, unsafe_allow_html=True)
        
        if len(filtered_comments) == 0:
            # nothing matched the filters
            st.info("No comments found matching your criteria.")
        else:
            comments_per_page = 5
            current_page = custom_pagination(
                len(filtered_comments),
                comments_per_page,
                st.session_state['explorer_page'],
                "explorer"
            )
            start_idx = (current_page - 1) * comments_per_page
            end_idx   = min(start_idx + comments_per_page, len(filtered_comments))

            for i, (comment_idx, row) in enumerate(
                    filtered_comments.iloc[start_idx:end_idx].iterrows(),
                    start=start_idx + 1
                ):
                # highlight if we have a search term
                full_text = highlight_keywords(str(row['Comment']), search_term) if search_term else str(row['Comment'])
                badge = format_sentiment_badge(row['Sentiment'])
            
                
                # Create the star button with current favorite state
                col1, col2 = st.columns([0.95, 0.05])
                
                with col1:
                    st.markdown(f"""
                    <div class='comment-box'>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                            <div>
                                <strong>Comment {i}</strong>
                                &nbsp;|&nbsp;
                                <span style="color:#6B7280;">{row['Topic_Label']}</span>
                            </div>
                            {badge}
                        </div>
                        <div>{full_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Check if this comment is already a favorite
                    is_favorite = comment_idx in st.session_state.get('favorite_comments', set())
                    star_icon = "⭐" if is_favorite else "☆"
                    
                    # Place the star button next to the comment box
                    if st.button(star_icon, key=f"star_{comment_idx}", help="Add to favourites"):
                        if is_favorite:
                            st.session_state['favorite_comments'].remove(comment_idx)
                        else:
                            st.session_state['favorite_comments'].add(comment_idx)
                        st.rerun()
                        
    # ⭐ Favorites page
    elif page == "⭐ Favourites":
                            st.header("Favourite Comments")
                    
                            # If no favorites yet
                            if not st.session_state['favorite_comments']:
                                st.info("You haven't saved any favourite comments yet. Star comments in the Comment Explorer to add them here.")
                                return
                    
                            # Gather your favorites
                            favorite_indices = list(st.session_state['favorite_comments'])
                            favorited_comments = df_comments.loc[favorite_indices]
                    
                            st.markdown(f"### You have {len(favorited_comments)} favourite comments")
                    
                            # Pagination state
                            if 'favorites_page' not in st.session_state:
                                st.session_state['favorites_page'] = 1
                    
                            comments_per_page = 5
                            current_page = custom_pagination(
                                len(favorited_comments),
                                comments_per_page,
                                st.session_state['favorites_page'],
                                "favorites"
                            )
                    
                            start_idx = (current_page - 1) * comments_per_page
                            end_idx = start_idx + comments_per_page
                    
                            # Display each favorited comment
                            for display_i, (idx, row) in enumerate(
                                    favorited_comments.iloc[start_idx:end_idx].iterrows(),
                                    start=start_idx + 1
                                ):
                                badge = format_sentiment_badge(row['Sentiment'])
                                st.markdown(f"""
                                    <div class='comment-box'>
                                      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                                        <div>
                                          <strong>Comment {idx}</strong> | 
                                          <span style="color:#6B7280;">{row['Topic_Label']}</span>
                                        </div>
                                        <div style="display:flex; align-items:center; gap:0.5rem;">{badge}</div>
                                      </div>
                                      <div>{row['Comment']}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                                # “Unfavorite” button
                                if st.button("★ Remove", key=f"unfav_{idx}"):
                                    st.session_state['favorite_comments'].remove(idx)
                                    st.rerun()

    # --- About page ---
    elif page == "📖 About":
        st.header("About This Study")
    
        # 1. Introduction
        st.subheader("Introduction")
        st.markdown("""
        This dashboard explores public perceptions of bi-directional charging—primarily vehicle-to-grid (V2G), 
        but also vehicle-to-home (V2H) and vehicle-to-load (V2L). Using topic modelling and sentiment analysis 
        on Reddit comments, we uncover key concerns and preferences around these technologies to inform 
        ePowerMove stakeholders.  
    
        For full project details, click the **ePowerMove** link in the top-right corner of this app.
        """)
    
        # 2. Methodology Overview
        st.subheader("2. Methodology Overview")
        st.markdown("""
        The methodology for this study comprises **data extraction** and **text analysis** phases.
        Reddit was chosen as the primary data source due to its rich ecosystem of user-generated
        content and community-driven discussions, organised into topic-specific subreddits where users
        self-select into interest-based communities. Only English-language texts were included to ensure consistency.
        """)
    
        st.markdown("### 2.1 Subreddit Selection")
        st.markdown("""
        - We used **PRAW** (v7.7.1) to interface with the Reddit API, authenticating with a client ID,
          secret, and user agent as per Reddit’s guidelines.
        - An extensive search query of 24 keywords (e.g. “bidirectional charging”, “V2G”, “vehicle-to-home”)
          was applied across 33 EV- and energy-focused subreddits (r/electricvehicles, r/teslamotors, r/solar, etc.).
        - No timeframe restriction was applied, capturing discussions up to **March 12, 2025**.
        """)
    
        st.markdown("### 2.2 Data Extraction")
        st.markdown("""
        - For each subreddit, posts matching our query were retrieved using **new**, **hot**, and **top** sorts.
        - Saved attributes: subreddit name, post ID, title, score, URL, comment count, timestamp, and text.
        - Removed duplicates by post ID → **780 unique posts**, then extracted all their comments.
        - After deduplication by comment ID → **17,430 unique comments**.
        """)
    
        st.markdown("### 2.3 Preprocessing")
        st.markdown("""
        - Removed null/empty/deleted comments, and filtered out posts with <5 words or generic “thanks” messages.
        - Stripped boilerplate (bot signatures, URLs, emojis, markdown, quotes) with ~23 regex patterns.
        - Final cleanup and de-duplication yielded **11,454 high-quality comments**.
        """)
    
        st.markdown("### 2.4 Topic Modelling")
        st.markdown("""
        - Utilised **BERTopic** with OpenAI’s `text-embedding-3-large` for transformer embeddings.
        - Dimensionality reduction via **UMAP** (_n_components_=10, _n_neighbors_=13, _min_dist_=0.0).
        - Clustering with **K-means** (50 clusters after experimentation).
        - Keywords selected via **Maximal Marginal Relevance** (diversity=0.5), then labelled with GPT-4o-mini.
        - Hierarchical modelling consolidated similar topics for clarity.
        """)
    
        st.markdown("### 2.5 Sentiment Analysis")
        st.markdown("""
        - Built a **gold standard** by manually annotating 600 stratified comments (200 per class).
        - Benchmarked four premium and four budget LLMs zero-shot; selected the best for full-dataset inference.
        - Prompts followed best-practice engineering guidelines, instructing the model as a “sentiment analysis expert.”
        """)

st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
