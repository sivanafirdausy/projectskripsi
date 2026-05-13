from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =========================
# LOAD DATA
# =========================
kbf_data = pd.read_csv('kbf_preprocessed.csv')
cbf_data = pd.read_csv('cbf_preprocessed.csv')

# =========================
# DISKRETISASI RATING (1–5)
# =========================
kbf_data['rating_class'] = np.floor(kbf_data['rating']).astype(int)
kbf_data['rating_class'] = kbf_data['rating_class'].clip(1, 5)

# =========================
# TF-IDF (HITUNG SEKALI)
# =========================
stopwords_id = [
    'di','ke','dari','pada','dalam','untuk','bagi',
    'dan','atau','tetapi','namun','bahwa','karena','sehingga',
    'aku','kamu','dia','mereka','kita','kami','saya',
    'nya','ini','itu','tersebut','yaitu','ialah','adalah',
    'juga','pun','saja','sangat','amat','terlalu'
]

cbf_data['text_final'] = cbf_data['text_final'].fillna('')

tfidf = TfidfVectorizer(
    stop_words=stopwords_id,
    lowercase=True,
    token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'
)

tfidf_matrix = tfidf.fit_transform(cbf_data['text_final'])

# =========================
# FUNGSI TARGET HARGA
# =========================
def get_target_harga(harga_range):
    if harga_range == 'rendah':
        return 25000
    elif harga_range == 'menengah':
        return 75000
    elif harga_range == 'tinggi':
        return 150000
    else:
        return 75000  # default

# =========================
# SIMILARITY HARGA
# =========================
def harga_similarity(harga, target):
    return 1 - min(abs(harga - target) / target, 1)

# =========================
# KBF FUNCTION (UPDATED)
# =========================
def kbf(df, pref):
    scores = []

    w = {
        'kategori': 0.7,
        'harga': 0.15,
        'rating': 0.15
    }

    for _, r in df.iterrows():
        s, t = 0, 0

        # Kategori
        if str(r['kategori']).lower() == pref['kategori'].lower():
            s += w['kategori']
        t += w['kategori']

        # Harga (SIMILARITY)
        sim_harga = harga_similarity(r['harga'], pref['target_harga'])
        s += w['harga'] * sim_harga
        t += w['harga']

        # Rating
        sim_rating = 1 - (abs(r['rating_class'] - pref['rating_class']) / 4)
        s += w['rating'] * sim_rating
        t += w['rating']

        scores.append(s / t)

    return scores

# =========================
# ROUTE HALAMAN
# =========================
@app.route('/')
def index():
    kategori_list = sorted(kbf_data['kategori'].dropna().unique())

    return render_template(
        'index.html',
        kategori_list=kategori_list,
    )

# =========================
# ROUTE RECOMMENDATION
# =========================
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.form

    try:
        # ---------- INPUT ----------
        kategori = data.get('kategori', '').strip()
        harga_range = data.get('harga_range', '').strip()
        rating_class = int(data.get('rating', 1))
        deskripsi = data.get('deskripsi', '').strip()

        if deskripsi == '':
            deskripsi = 'lipstick'

        # ---------- TARGET HARGA ----------
        target_harga = get_target_harga(harga_range)

        # ---------- CANDIDATE SELECTION (TOP-N TERDEKAT) ----------
        df_temp = kbf_data.copy()
        df_temp['harga_diff'] = abs(df_temp['harga'] - target_harga)

        df_filtered = df_temp.sort_values('harga_diff').head(200).copy()

        pref = {
            'kategori': kategori,
            'rating_class': rating_class,
            'target_harga': target_harga
        }

        # ---------- KBF ----------
        df_filtered['kbf_score'] = kbf(df_filtered, pref)

        # ---------- CBF ----------
        user_vec = tfidf.transform([deskripsi])
        cbf_score = cosine_similarity(user_vec, tfidf_matrix).flatten()

        cbf_temp = cbf_data.copy()
        cbf_temp['cbf_score'] = cbf_score

        # ---------- HYBRID ----------
        hybrid = pd.merge(
            cbf_temp[['nama_produk','nama_shade','cbf_score']],
            df_filtered[['nama_produk','nama_shade','kbf_score','brand']],
            on=['nama_produk','nama_shade'],
            how='left'
        )

        hybrid['brand'] = hybrid['brand'].fillna('-')
        hybrid['kbf_score'] = hybrid['kbf_score'].fillna(0)

        hybrid['hybrid_score'] = (
            0.8 * hybrid['kbf_score'] +
            0.2 * hybrid['cbf_score']
        )

        # Urutkan dari skor tertinggi
        hybrid_sorted = hybrid.sort_values('hybrid_score', ascending=False)

        # Ambil 1 shade terbaik per produk
        hybrid_unique = hybrid_sorted.drop_duplicates(subset=['nama_produk'], keep='first')

        # Ambil top 5 produk
        hybrid_top = hybrid_unique.head(5)

        # ================= LOWEST HYBRID (BOTTOM 5) =================
        hybrid_bottom = hybrid_unique.sort_values(
            'hybrid_score',
            ascending=True
        ).head(5)

        return jsonify({
            'hybrid': hybrid_top[[
                'nama_produk','nama_shade','brand',
                'kbf_score','cbf_score','hybrid_score'
            ]].to_dict(orient='records'),

            'not_recommended': hybrid_bottom[[
                'nama_produk','nama_shade',
                'kbf_score','cbf_score','hybrid_score'
            ]].to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)