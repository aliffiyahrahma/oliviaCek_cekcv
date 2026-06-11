import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))  # Load environment variables from .env file

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, send_file, jsonify
from concurrent.futures import ThreadPoolExecutor
from authlib.integrations.flask_client import OAuth
from collections import Counter
import re, ast, uuid, json, pymysql.cursors, os, torch, io, zipfile, markdown, nltk, PyPDF2, pdfplumber
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from tfidf import tfidf
from gemini import gemini
from bert import bert
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from pprint import pprint
from functools import wraps
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from keybert import KeyBERT
from langdetect import detect
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Inisialisasi model BERT   
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
model = AutoModel.from_pretrained('bert-base-multilingual-cased')

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='cek_cv',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
app = Flask(__name__)


app.config["SECRET_KEY"] = os .environ.get("SECRET_KEY")
app.config['GOOGLE_CLIENT_ID'] = os .environ.get("GOOGLE_CLIENT_ID")
app.config['GOOGLE_CLIENT_SECRET'] = os .environ.get("GOOGLE_CLIENT_SECRET")
app.config['GOOGLE_DISCOVERY_URL'] = os .environ.get("GOOGLE_DISCOVERY_URL")

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs',  # Tambahkan URL jwks_uri
    client_kwargs={
        'scope': 'openid email profile'
    }
)

nltk.data.path.append('C:/nltk_data')
nltk.download('punkt')

# Tentukan direktori untuk menyimpan file yang diunggah sementara (CV)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Inisialisasi KeyBERT
kw_model = KeyBERT()

SCAN_RESULTS_CACHE = {} 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk ekstraksi teks dari PDF
def extract_text_from_pdf(pdf_file_path):
    try:
        text = ""
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Terjadi kesalahan saat ekstraksi: {e}")
        return None

# Fungsi untuk mendapatkan embedding dari teks
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Fungsi untuk menghitung kesamaan CV & JobDes menggunakan cosine similarity akurasi angka
def calculate_similarity(cv_text, jd_embedding):
    cv_embedding = get_embedding(cv_text)
    similarity_score = cosine_similarity(cv_embedding, jd_embedding).flatten()
    return similarity_score[0]

CUSTOM_STOPWORDS = set([
    'dan', 'di', 'dari', 'yang', 'untuk', 'dengan', 'dalam', 'pada', 'atau', 'sebagai', 'adalah', 
    'ini', 'itu', 'ke', 'oleh', 'juga', 'agar', 'dapat', 'lebih', 'mampu', 'memiliki',
    'menggunakan', 'melakukan', 'membuat', 'selanjutnya'
])

def extract_keywords(text, top_n=15):
    text_cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = [
        word for word in text_cleaned.split()
        if word not in ENGLISH_STOP_WORDS and word not in CUSTOM_STOPWORDS and len(word) > 4
    ]
    text_filtered = " ".join(words)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    tfidf_vectorizer.fit([text_filtered])
    tfidf_keywords = set(tfidf_vectorizer.get_feature_names_out())

    keybert_result = kw_model.extract_keywords(
        text_filtered, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n
    )
    keybert_keywords = set([kw[0] for kw in keybert_result if len(kw[0]) > 4])

    combined_keywords = list(tfidf_keywords.union(keybert_keywords))
    final_keywords = [kw for kw in combined_keywords if len(kw.split()) > 1 or len(kw) > 6]
    return final_keywords

#fungsi utama untuk menganalisis CV dan menghitung skor kesesuaian total
def process_cv(cv_path, jd_embedding, jd_text, keywords):
    cv_text = extract_text_from_pdf(cv_path)
    if cv_text:
        similarity_score = calculate_similarity(cv_text, jd_embedding)
        adjusted_score = adjust_score(similarity_score, keywords, cv_text, jd_text)
        return adjusted_score, cv_text
    return 0, ""

def get_language_penalty(jd_text, cv_text):
    try:
        jd_lang = detect(jd_text)
        cv_lang = detect(cv_text)
        return -0.3 if jd_lang != cv_lang else 0
    except:
        return 0

# Fungsi untuk menyesuaikan skor berdasarkan jumlah kata kunci yang ditemukan
def adjust_score(similarity_score, jd_keywords, cv_text, jd_text):
    cv_text_lower = cv_text.lower()

    matched_keywords = [
        kw for kw in jd_keywords
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', cv_text_lower)
    ]
    match_ratio = len(matched_keywords) / len(jd_keywords) if jd_keywords else 0

    bonus = 0.4 * match_ratio  # max bonus 0.4
    penalty = get_language_penalty(jd_text, cv_text)

    adjusted = (0.6 * similarity_score) + bonus + penalty + 0.2

    # Skor dibatasi antara 0 sampai 95 (biar tidak terlalu tinggi)
    return min(max(adjusted, 0), 99.0)

def highlight_text(cv_text, keywords):
    highlighted_text = cv_text
    for keyword in keywords:
        highlighted_text = re.sub(
            rf'({re.escape(keyword)})', 
            r'<span style="background-color: yellow">\1</span>', 
            highlighted_text, 
            flags=re.IGNORECASE
        )
    return highlighted_text

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and t not in CUSTOM_STOPWORDS]
    return " ".join(tokens)





@app.route('/uploaded_files/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

#pecah text jadi 3 bagian
def extract_text_sections(full_text, line_count=3):
    lines = full_text.split('\n')
    total_lines = len(lines)
    head = '\n'.join(lines[:line_count])
    middle = '\n'.join(lines[line_count:line_count*2])
    tail = '\n'.join(lines[-line_count:])
    return head, middle, tail

def detect_spam_keywords(text, threshold=15):
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    spam_words = {word: count for word, count in word_counts.items() if count > threshold}
    return spam_words

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if 'user_id' not in session:
        flash("Please login to access this feature", "danger")
        return redirect(url_for('login'))

    user_id = session['user_id']
    pricing_status = session.get('pricing', 'free')  # Default ke 'free' jika tidak ada data

    # Jika pengguna memiliki akses bebas (pricing = 'plus' atau 'pro'), abaikan pengecekan scan_count
    if pricing_status not in ['plus', 'pro']:
        # Cek jumlah scan dari database
        try:
            with connection.cursor() as cursor:
                sql = "SELECT COUNT(*) AS scan_count FROM history WHERE user_id = %s"
                cursor.execute(sql, (user_id,))
                result = cursor.fetchone()
                scan_count = result['scan_count'] if result else 0
        except Exception as e:
            flash(f"Failed to fetch scan count: {e}", "danger")
            return render_template('scan.html', scan_disabled=True)

        # Jika sudah 10 kali scan, disable fitur scan
        if scan_count >= 5:
            return render_template('scan.html', scan_disabled=True)

    if request.method == 'POST':
        # Upload CV files
        cv_files = request.files.getlist('cv[]')
        job_description_file = request.files['job_description']
        spam_detected_files = []
        original_cv_texts = {}
        cv_spam_keyword_map = {}


        saved_cv_paths = []
        for cv_file in cv_files:
            if cv_file and allowed_file(cv_file.filename):
                cv_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(cv_file.filename))
                cv_file.save(cv_path)
                saved_cv_paths.append(cv_path)

                text = extract_text_from_pdf(cv_path)
                original_cv_texts[cv_path] = text

                spam_words = detect_spam_keywords(text)
                if spam_words:
                    cv_spam_keyword_map[cv_path] = spam_words
                else:
                    cv_spam_keyword_map[cv_path] = {}    


        if job_description_file and allowed_file(job_description_file.filename):
            jd_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(job_description_file.filename))
            job_description_file.save(jd_path)

            # Ekstraksi teks asli dari Job Description
            jd_text = extract_text_from_pdf(jd_path)
            jd_embedding = get_embedding(jd_text)
            keywords = extract_keywords(jd_text)

            # Analisis CV menggunakan ThreadPoolExecutor
            similarities = {}
            highlighted_cv_texts = {}
            original_cv_texts = {}
            cv_addresses = {}
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(process_cv, cv_path, jd_embedding, jd_text, keywords): cv_path
                    for cv_path in saved_cv_paths
                }
                for future in futures:
                    cv_path = futures[future]
                    try:
                        similarity, original_text = future.result()
                        similarities[cv_path] = similarity
                        highlighted_cv_texts[cv_path] = highlight_text(original_text, keywords)
                        original_cv_texts[cv_path] = original_text
                    except Exception as exc:
                        print(f"{cv_path} generated an exception: {exc}")

            # Urutkan hasil berdasarkan skor kesamaan (dari terbesar ke terkecil)
            sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

            # Ambil tiga CV terbaik
            top_3_results = sorted_results[:3]

            top_3_data = []
            for cv_path, similarity_score in top_3_results:
                original_text = original_cv_texts[cv_path]
                head, middle, tail = extract_text_sections(original_text, line_count=6)

                top_3_data.append({
                    "cv_path": cv_path,
                    "similarity_score_percent": similarity_score * 100,
                    "cv_address": cv_addresses.get(cv_path, "Alamat tidak ditemukan"),
                    "cv_head": head.replace('\n', '<br>'),
                    "cv_middle": middle.replace('\n', '<br>'),
                    "cv_tail": tail.replace('\n', '<br>'),
                    "cv_text": original_text,  # <-- Diperlukan untuk simpan ke DB
                    "keywords": keywords
                })

            feedback_section = []
            for cv_path in saved_cv_paths:
                filename = os.path.basename(cv_path)
                original_text = original_cv_texts[cv_path]
                highlighted_text = highlight_text(original_text, keywords)
                feedback_points, total_score = generate_feedback_detailed(original_text, jd_text)
                
                feedback_section.append({
                    "cv_path": cv_path,
                    "cv_pdf_url": url_for('serve_uploaded_file', filename=os.path.basename(cv_path)),
                    "score": total_score,
                    "feedback": feedback_points,
                     "filename": filename,
                     "spam_keywords": cv_spam_keyword_map.get(cv_path, {})
                })


            # Simpan data ke cache server-side
            scan_id = str(uuid.uuid4())
            SCAN_RESULTS_CACHE[scan_id] = {
                'top_3_data': top_3_data,
                'job_description': jd_text
            }
            # Store ONLY the ID in the session
            session['last_scan_id'] = scan_id

            # Simpan ke database hanya jika ada hasil
            if top_3_data:
                try:
                    with connection.cursor() as cursor:
                        sql = """
                            INSERT INTO history (job_description, best_cv_path, similarity_score, user_id, best_cv_text, best_cv_path2, similarity_score2, best_cv_text2, best_cv_path3, similarity_score3, best_cv_text3)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(sql, (jd_text, top_3_data[0]['cv_path'], top_3_data[0]['similarity_score_percent'], user_id, top_3_data[0]['cv_text'], top_3_data[1]['cv_path'], top_3_data[1]['similarity_score_percent'], top_3_data[1]['cv_text'], top_3_data[2]['cv_path'], top_3_data[2]['similarity_score_percent'], top_3_data[2]['cv_text']))
                        connection.commit()
                    flash("Scan completed successfully!")
                except Exception as e:
                    flash(f"Failed to save scan result: {e}", "danger")

            return render_template('result.html', top_3_data=top_3_data, job_desc=jd_text, feedback_section=feedback_section)
        
    return render_template('scan.html', scan_disabled=False)



INDONESIAN_CITIES = {
    "jakarta", "bandung", "surabaya", "yogyakarta", "medan", "semarang",
    "makassar", "palembang", "depok", "tangerang", "bekasi", "bogor",
    "malang", "denpasar", "balikpapan", "pekanbaru", "padang", "lampung",
    "bengkulu", "samarinda", "manado", "pontianak", "aceh", "cirebon",
    "solo", "purwokerto", "banda aceh", "kendari", "gorontalo", "lombok"
}

INDONESIAN_PROVINCES = {
    "dki jakarta", "jawa barat", "jawa tengah", "jawa timur", "daerah istimewa yogyakarta",
    "sumatera utara", "sulawesi selatan", "sumatera selatan", "banten", "riau",
    "kalimantan timur", "bali", "sumatera barat", "lampung", "bengkulu",
    "aceh", "papua", "sulawesi utara", "kalimantan barat", "ntb", "ntt",
    "kepulauan riau", "bangka belitung", "jambi", "gorontalo", "maluku",
    "kalimantan selatan", "kalimantan tengah", "sulawesi tengah", "sulawesi tenggara"
}

# --- Fungsi untuk Ekstraksi Alamat yang Lebih Baik ---
def extract_address_from_text(cv_text):
    found_elements = {
        'postal_code': None,
        'city': None,
        'province': None,
        'full_address_snippet': None # Potongan alamat lengkap jika ditemukan
    }
    
    # 1. Cari Kode Pos (5 digit angka)
    # Ini adalah yang paling andal karena polanya jelas
    postal_code_pattern = re.compile(r'\b(\d{5})\b')
    match_postal = postal_code_pattern.search(cv_text)
    if match_postal:
        found_elements['postal_code'] = match_postal.group(1)

    # Normalisasi teks untuk pencarian kota/provinsi
    normalized_cv_text = cv_text.lower().replace('.', '').replace(',', '').replace('(', '').replace(')', '')
    
    # 2. Cari Provinsi dan Kota
    # Prioritaskan pencarian dari yang lebih panjang (provinsi multi-kata)

    # Cari Provinsi
    found_province = None
    for province in sorted(INDONESIAN_PROVINCES, key=len, reverse=True):
        if re.search(r'\b' + re.escape(province) + r'\b', normalized_cv_text):
            found_province = province.title()
            break
    if found_province:
        found_elements['province'] = found_province

    # Cari Kota
    found_city = None
    for city in sorted(INDONESIAN_CITIES, key=len, reverse=True):
        if re.search(r'\b' + re.escape(city) + r'\b', normalized_cv_text):
            # Hindari kota yang merupakan bagian dari provinsi jika provinsi sudah ditemukan
            # Contoh: jika "Jawa Barat" ditemukan, jangan ambil "Bandung" dari "Jawa Barat"
            if not (found_province and city in found_province.lower() and city != found_province.lower()):
                found_city = city.title()
                break
    if found_city:
        found_elements['city'] = found_city

    # 3. Coba dapatkan potongan alamat lengkap (opsional, untuk tampilan lebih detail)
    # Ini adalah versi perbaikan dari regex Anda sebelumnya
    full_address_pattern = re.compile(
        r'(?:jl\.|jalan|perum|komplek)\s*[\w\s\d\-\.]+(?:\s+no\.\s*\d+)?' + # Jalan / No.
        r'(?:\s*rt\s*\d+\s*rw\s*\d+)?' +                                  # RT/RW
        r'(?:\s*kel\.\s*[\w\s\d\.]+)?' +                                  # Kelurahan
        r'(?:\s*kec\.\s*[\w\s\d\.]+)?' +                                  # Kecamatan
        r'(?:\s*kota\s*[\w\s\d\.]+)?' +                                   # Kota
        r'(?:\s*prov\.\s*[\w\s\d\.]+)?' +                                 # Provinsi
        r'(?:\s*\d{5})?',                                                 # Kode Pos
        re.IGNORECASE | re.DOTALL # re.DOTALL untuk mencocokkan newline
    )
    
    # Cari di sekitar keyword alamat dulu
    lines = cv_text.split('\n')
    address_keywords_in_text = ["alamat:", "address:", "tinggal:", "domisili:"]
    for i, line in enumerate(lines):
        for keyword in address_keywords_in_text:
            if keyword in line.lower():
                # Ambil blok teks setelah keyword, mungkin beberapa baris
                block_start = max(0, i - 1) # Sedikit ke atas jika alamatnya dimulai sebelum keyword
                block_end = min(len(lines), i + 4) # Beberapa baris ke bawah
                text_block = "\n".join(lines[block_start:block_end])
                
                match_full_address = full_address_pattern.search(text_block)
                if match_full_address:
                    found_elements['full_address_snippet'] = match_full_address.group(0).strip()
                    break # Keluar dari loop keyword
        if found_elements['full_address_snippet']:
            break # Keluar dari loop baris
            
    # Jika tidak ditemukan di sekitar keyword, coba seluruh teks
    if not found_elements['full_address_snippet']:
        match_full_address = full_address_pattern.search(cv_text)
        if match_full_address:
            found_elements['full_address_snippet'] = match_full_address.group(0).strip()

    # 4. Gabungkan hasil yang ditemukan
    parts = []
    if found_elements['full_address_snippet']:
        # Jika ada potongan alamat lengkap, prioritaskan itu
        # Dan coba pastikan kota/provinsi yang terdeteksi ada di dalamnya
        snippet = found_elements['full_address_snippet']
        if found_elements['city'] and found_elements['city'].lower() not in snippet.lower():
            parts.append(found_elements['city'])
        if found_elements['province'] and found_elements['province'].lower() not in snippet.lower():
            parts.append(found_elements['province'])
        
        # Tambahkan cuplikan, batasi panjangnya
        display_snippet = snippet
        if len(display_snippet) > 150: # Batasi panjang untuk display
            display_snippet = display_snippet[:150] + "..."
        parts.append(display_snippet)

    else:
        # Jika tidak ada cuplikan lengkap, gabungkan yang ditemukan
        if found_elements['city']:
            parts.append(found_elements['city'])
        if found_elements['province']:
            parts.append(found_elements['province'])
        if found_elements['postal_code']:
            parts.append(f"Kode Pos: {found_elements['postal_code']}")
            
    if not parts:
        return "Alamat tidak ditemukan"
    
    # Gabungkan dengan preferensi Kota, Provinsi, Kode Pos, atau snippet
    final_address = ", ".join(filter(None, [
        found_elements['city'],
        found_elements['province'],
        found_elements['postal_code'],
        found_elements['full_address_snippet'] if not (found_elements['city'] or found_elements['province'] or found_elements['postal_code']) else None
    ]))
    
    # Batasi panjang total untuk tampilan di PDF
    if len(final_address) > 150:
        return final_address[:150] + "..."
        
    return final_address if final_address else "Alamat tidak ditemukan"

#print pdffffffffffff
@app.route('/generate_print_pdf', methods=['POST'])
def generate_print_pdf():
    if 'user_id' not in session:
        flash("Please login to access this feature", "danger")
        return redirect(url_for('login'))

    # Ambil ID hasil scan
    scan_id = session.get('last_scan_id')
    
    if not scan_id or scan_id not in SCAN_RESULTS_CACHE:
        flash("Tidak ada data hasil scan terbaru untuk dicetak. Silakan lakukan scan ulang.", "warning")
        return redirect(url_for('scan'))

    scan_data = SCAN_RESULTS_CACHE[scan_id]
    top_3_data = scan_data['top_3_data']
    job_desc = scan_data['job_description']

    # Nama file PDF
    pdf_filename = f"Laporan_Scan_CV_{session['user_id']}_{scan_id}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Tambah style justify
    justified_style = ParagraphStyle(
        'Justify',
        parent=styles['Normal'],
        alignment=4,  # TA_JUSTIFY
        leading=14
    )

    title_style = ParagraphStyle(
        'Title',
        parent=styles['h1'],
        fontSize=20,
        alignment=1,  # TA_CENTER
        spaceAfter=14
    )
    story.append(Paragraph("Laporan Hasil Scan CV", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Deskripsi Pekerjaan:</b>", styles['h3']))
    story.append(Paragraph(job_desc, justified_style))
    story.append(Spacer(1, 0.4 * inch))

    story.append(Paragraph("<b>Daftar Kandidat Terbaik:</b>", styles['h3']))
    story.append(Spacer(1, 0.2 * inch))

    data = [['Nama CV', 'Skor Kecocokan']]
    for result in top_3_data:
        cv_name = os.path.basename(result['cv_path'])
        score = f"{result['similarity_score_percent']:.2f}%"
        data.append([cv_name, score])

    table = Table(data, colWidths=[4*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5 * inch))

    intro_paragraph = Paragraph(
        "Laporan ini disusun sebagai bagian dari proses evaluasi dan seleksi kandidat berdasarkan tingkat kecocokan antara dokumen Curriculum Vitae (CV) masing-masing pelamar dengan deskripsi pekerjaan yang telah ditentukan. "
        "Melalui sistem pemindaian otomatis berbasis kecerdasan buatan, setiap CV dianalisis secara menyeluruh untuk mengidentifikasi relevansi isi terhadap kualifikasi, pengalaman, dan kompetensi yang dibutuhkan oleh posisi pekerjaan tersebut. "
        "Proses ini dilakukan secara objektif dan terukur, dengan mempertimbangkan berbagai aspek penting seperti kata kunci yang sesuai, struktur dokumen, serta pengalaman kerja yang relevan. "
        "Berikut ini merupakan tiga kandidat teratas yang memiliki tingkat kecocokan tertinggi dengan posisi yang dibutuhkan berdasarkan hasil analisis sistem.",
        justified_style
    )
    story.append(intro_paragraph)
    story.append(Spacer(1, 0.3 * inch))

    try:
        doc.build(story)
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename, mimetype='application/pdf')
    except Exception as e:
        flash(f"Gagal membuat PDF: {e}", "danger")
        return redirect(url_for('scan'))
    
@app.route("/history")
def history():
    user_id = session.get('user_id')  # Mengambil user_id dari sesi, jika tidak ada, nilainya None
    results = []

    if user_id:
        try:
            with connection.cursor() as cursor:
                # Query untuk mengambil jumlah scan per tanggal, dikelompokkan berdasarkan user dan tanggal
                sql = """
                    SELECT DATE(created_at) AS scan_date, COUNT(*) AS total_scan
                    FROM history
                    WHERE user_id = %s
                    GROUP BY DATE(created_at)
                    ORDER BY scan_date DESC
                """
                cursor.execute(sql, (user_id,))
                results = cursor.fetchall()  # Mengambil semua hasil query sebagai list of dicts
        except Exception as e:
            flash(f"Error retrieving scan history: {e}", "danger")
    
    return render_template("history.html", results=results)

@app.route("/feedhistory")
def feedhistory():
    user_id = session.get('user_id')  # Mengambil user_id dari sesi, jika tidak ada, nilainya None
    results = []

    if user_id:
        try:
            with connection.cursor() as cursor:
                # Query untuk mengambil jumlah scan per tanggal, dikelompokkan berdasarkan user dan tanggal
                sql = """
                    SELECT DATE(created_at) AS scan_date, cv_text, feedback, score
                    FROM feedback
                    WHERE user_id = %s
                    ORDER BY scan_date DESC
                """
                cursor.execute(sql, (user_id,))
                results = cursor.fetchall()  # Mengambil semua hasil query sebagai list of dicts

                 # Parse kolom feedback JSON string ke Python list/dict
                for row in results:
                    try:
                        # Ubah dari string Python literal ke list of dict
                        row['feedback'] = ast.literal_eval(row['feedback'])
                    except Exception:
                        row['feedback'] = []
        except Exception as e:
            flash(f"Error retrieving scan history: {e}", "danger")
    
    return render_template("feedhistory.html", results=results)
    
#fedbackkkkkkkkkkk
def generate_feedback_detailed(cv_text, jd_text):
    jd_keywords = extract_keywords(jd_text, top_n=20)
    cv_keywords = extract_keywords(cv_text, top_n=20)

    missing_keywords = [kw for kw in jd_keywords if kw.lower() not in cv_text.lower()]

    similarity_score = calculate_similarity(cv_text, get_embedding(jd_text))
    total_score = adjust_score(similarity_score, jd_keywords, cv_text, jd_text) * 100

    feedback_points = []

    if missing_keywords:
        feedback_points.append({
            "judul": "Kata Kunci Teknis Tidak Ditemukan",
            "saran": (
                "Beberapa istilah penting dari deskripsi pekerjaan tidak ditemukan di CV Anda. "
                "Ini bisa membuat sistem atau HR menganggap Anda kurang relevan dengan posisi ini.\n"
                # f"Kata kunci yang perlu ditambahkan: {', '.join(missing_keywords)}"
            )
        })
    else:
        feedback_points.append({
            "judul": "Kecocokan Kata Kunci Teknis",
            "saran": "CV Anda sudah mengandung sebagian besar kata kunci teknis yang dicari berdasarkan deskripsi pekerjaan. Bagus!"
        })

    if not any(word in cv_text.lower() for word in ['pengalaman', 'pendidikan', 'skill', 'keahlian']):
        feedback_points.append({
            "judul": "Struktur CV Kurang Lengkap",
            "saran": (
                "Pastikan CV Anda memiliki struktur dasar seperti:\n"
                "- Ringkasan profil\n- Pengalaman kerja\n- Pendidikan\n- Keahlian\n- Sertifikasi (jika ada)"
            )
        })

    if "tahun" not in cv_text.lower() and "bulan" not in cv_text.lower():
        feedback_points.append({
            "judul": "Detail Pengalaman Tidak Jelas",
            "saran": (
                "Tampilkan pengalaman kerja Anda dengan durasi (misalnya '2 tahun sebagai Web Developer'). "
                "Durasi penting untuk menilai tingkat pengalaman Anda."
            )
        })

    soft_skills = ['komunikasi', 'kerja tim', 'kepemimpinan', 'problem solving', 'adaptif', 'inisiatif']
    soft_skills_found = [skill for skill in soft_skills if skill in preprocess(cv_text)]
    if len(soft_skills_found) < 2:
        feedback_points.append({
            "judul": "Soft Skills Kurang Terlihat",
            "saran": (
                "Soft skills juga penting. Pertimbangkan untuk menambahkan poin tentang kemampuan seperti: "
                "komunikasi, kerja tim, adaptasi, atau kepemimpinan."
            )
        })

    informal_words = ['gue', 'aku', 'kamu', 'banget', 'nih']
    informal_found = [w for w in informal_words if w in cv_text.lower()]
    if informal_found:
        feedback_points.append({
            "judul": "Gunakan Bahasa Profesional",
            "saran": (
                "Beberapa kata yang digunakan terdengar terlalu informal untuk CV, seperti: "
                f"{', '.join(informal_found)}. Gunakan bahasa yang profesional dan netral."
            )
        })

    return feedback_points, round(total_score, 2)


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user_id' not in session:
        flash("Please login to access this feature", "danger")
        return redirect(url_for('login'))

    user_id = session['user_id']
    pricing_status = session.get('pricing', 'free')  # Default ke 'free' jika tidak ada data

    # Jika pengguna memiliki akses bebas (pricing = 'plus' atau 'pro'), abaikan pengecekan scan_count
    if pricing_status not in ['plus', 'pro']:
        # Cek jumlah scan dari database
        try:
            with connection.cursor() as cursor:
                sql = "SELECT COUNT(*) AS scan_count FROM feedback WHERE user_id = %s"
                cursor.execute(sql, (user_id,))
                result = cursor.fetchone()
                scan_count = result['scan_count'] if result else 0
        except Exception as e:
            flash(f"Failed to fetch scan count: {e}", "danger")
            return render_template('feedback.html', scan_disabled=True)

        # Jika sudah 10 kali scan, disable fitur scan
        if scan_count >= 5:
            return render_template('feedback.html', scan_disabled=True)

    if request.method == 'POST':
        cv_file = request.files['cv']
        jd_file = request.files['job_description']

        if not (cv_file and jd_file and allowed_file(cv_file.filename) and allowed_file(jd_file.filename)):
            flash("Kedua file harus diunggah dan dalam format PDF yang valid.", "danger")
            return redirect(request.url)

        cv_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(cv_file.filename))
        jd_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(jd_file.filename))
        cv_file.save(cv_path)
        jd_file.save(jd_path)

        cv_text = extract_text_from_pdf(cv_path)
        jd_text = extract_text_from_pdf(jd_path)

        # Hitung embedding JD sekali
        jd_embedding = get_embedding(jd_text)

        # Ekstraksi kata kunci JD
        keywords = extract_keywords(jd_text)

        # Sorot teks berdasarkan keywords
        highlighted_cv_text = highlight_text(cv_text, keywords)

        feedback_points, total_score = generate_feedback_detailed(cv_text, jd_text)
        spam_keywords = detect_spam_keywords(cv_text)

        # Simpan ke DB
        try:
            with connection.cursor() as cursor:
                sql = """
                    INSERT INTO feedback (user_id, cv_text, jobdesc, feedback, score)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    user_id,
                    highlighted_cv_text.replace('\n', '<br>'),
                    jd_text,
                    json.dumps(feedback_points),
                    total_score
                ))
                connection.commit()
        except Exception as e:
            flash(f"Gagal menyimpan feedback: {e}", "danger")
        cv_display_url = url_for('serve_uploaded_file', filename=os.path.basename(cv_path))


        return render_template("feedresult.html",
                               feedback=feedback_points,
                               score=total_score,
                               jd_text=jd_text,
                               cv_text=cv_text,
                             cv_pdf_url=cv_display_url ,  spam_keywords=spam_keywords )
        
    return render_template('feedback.html')

# Folder untuk menyimpan file bukti transfer
UPLOAD_FOLDER_PRICING = 'uploads_pricing'
ALLOWED_EXTENSIONS_PRICING = {'jpg', 'jpeg', 'png'}
MAX_CONTENT_LENGTH_PRICING = 2 * 1024 * 1024  # Maksimal ukuran file 2 MB

app.config['UPLOAD_FOLDER_PRICING'] = UPLOAD_FOLDER_PRICING
app.config['MAX_CONTENT_LENGTH_PRICING'] = MAX_CONTENT_LENGTH_PRICING

if not os.path.exists(UPLOAD_FOLDER_PRICING):
    os.makedirs(UPLOAD_FOLDER_PRICING)

def allowed_file_pricing(filename):
    """Periksa apakah file memiliki ekstensi yang diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_PRICING

@app.route('/pricing_plus', methods=['GET', 'POST'])
def pricing_plus():
    if 'user_id' not in session:
        if request.method == 'POST':
            return jsonify({"status": "error", "message": "Silakan login untuk melanjutkan"})
        else:
            flash("Silakan login untuk melanjutkan", "danger")
            return redirect(url_for('login'))

    if request.method == 'POST':
        pricing = request.form.get('pricing')
        status = 'paid'
        metode = request.form.get('metode')
        user_id = session.get('user_id')
        

        filename = None # Default value jika tidak ada file

        if not metode:
            return jsonify({"status": "error", "message": "Metode pembayaran tidak dipilih"})

        # Perbarui logika untuk menangani file yang opsional

        try:
            with connection.cursor() as cursor:
                # Simpan data transaksi pricing
                sql_insert = """
                    INSERT INTO pricing (pricing, user_id, status, metode)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql_insert, (pricing, user_id, status, metode))

                # Update kolom pricing di tabel user
                sql_update = "UPDATE users SET pricing = 'plus' WHERE id = %s"
                cursor.execute(sql_update, (user_id,))

                connection.commit()
                session['pricing'] = pricing

            return jsonify({"status": "paid"})
        
        except Exception as e:
            return jsonify({"status": "error", "message": f"Gagal menyimpan data: {e}"})

    return render_template("pricing_plus.html")

UPLOAD_FOLDER = os.path.join(app.root_path, app.config['UPLOAD_FOLDER_PRICING'])
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/uploads_pricing/<filename>')
def uploads_pricing(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER_PRICING'], filename)
    print(f"Accessing file: {file_path}")
    return send_from_directory(app.config['UPLOAD_FOLDER_PRICING'], filename)

# @app.route("/")
# def index():
#     return render_template("index.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/pricing")
def pricing():
    return render_template("pricing.html")

@app.route("/pricing_pro")
def pricing_pro():
    return render_template("pricing_pro.html")

@app.route('/send-contact', methods=['POST'])
def send_contact():
    try:
        nama = request.form['nama']
        email = request.form['email']
        teks = request.form['teks']

        # Simpan ke database
        with connection.cursor() as cursor:
            sql = "INSERT INTO messages (nama, email, teks) VALUES (%s, %s, %s)"
            cursor.execute(sql, (nama, email, teks))
            connection.commit()

        flash('Pesan berhasil dikirim!', 'success')
        return redirect(url_for('contact'))
    except Exception as e:
        flash(f'Gagal mengirimkan pesan: {e}', 'danger')
        return redirect(url_for('contact'))

@app.route("/login")
def login():
    return render_template("login.html")

@app.context_processor
def inject_user_role():
    return dict(user_role=session.get('role'))

@app.route('/req-login', methods=['POST'])
def req_login():
    email = request.form['email']
    password = request.form['password']

    # Cek apakah email dan password sesuai dengan superadmin
    if email == 'superadmin@gmail.com' and password == 'superadmin':
        session.clear()
        session['user_id'] = 'superadmin'
        session['pricing'] = 'Unlimited'
        session['username'] = 'Super Admin'
        session['nama'] = 'Superadmin'
        session['email'] = email
        session['role'] = 'Superadmin'
        flash('Login Superadmin berhasil!', 'success')
        return redirect(url_for('dashboard'))

    try:
        with connection.cursor() as cursor:
            # Cari user berdasarkan email
            sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(sql, (email,))
            user = cursor.fetchone()

            if user and check_password_hash(user['password'], password):
                # Jika password cocok
                session.clear()
                session['user_id'] = user['id']
                session['pricing'] = user['pricing']
                session['username'] = user['username']
                session['nama'] = user['nama']
                session['email'] = user['email']
                session['role'] = user['role']
                flash('Login berhasil!', 'success')

                if user['role'] == 'Admin':
                    return redirect(url_for('dashboard'))
                else:
                    return redirect(url_for('index'))
            else:
                flash('Email atau password salah', 'danger')
                return redirect(url_for('login'))
    except Exception as e:
        flash(f'Error saat login: {e}', 'danger')
        return redirect(url_for('login'))


@app.route('/login-google')
def login_google():
    redirect_uri = url_for('authorize_google', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/callback')
def authorize_google():
    try:
        token = google.authorize_access_token()
        user_info = google.get('userinfo').json()
    except Exception as e:
        flash(f'Login dengan Google gagal. {e}', 'danger')
        print(f"Google Auth Error: {e}")
        return redirect(url_for('login'))

    if user_info:
        email = user_info['email']
        name = user_info.get('name')
        username = email.split('@')[0] # Generate username dari email

        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

            if user:
                # User sudah terdaftar, langsung login
                session.clear()
                session['user_id'] = user['id']
                session['nama'] = name
                session['email'] = email
                session['username'] = username
                session['role'] = user['role']
                session['pricing'] = user['pricing']
                flash('Login berhasil!', 'success')
                return redirect(url_for('index'))
            else:
                # User baru, tampilkan modal pilihan role
                # Redirect kembali ke halaman register (atau halaman lain) dengan data user sebagai parameter URL
                # Ini akan memicu JS di halaman register untuk menampilkan modal
                return redirect(url_for('register',
                                        show_role_modal='true',
                                        email=email,
                                        name=name,
                                        username=username,
                                        is_new_user='true')) # Tambahkan indikator user baru
    else:
        flash('Login dengan Google gagal: Informasi user tidak ditemukan.', 'danger')
        return redirect(url_for('login'))


@app.route('/process-google-role', methods=['POST'])
def process_google_role():
    email = request.form.get('email')
    name = request.form.get('name')
    username = request.form.get('username')
    role = request.form.get('role')
    is_new_user_str = request.form.get('is_new_user')
    is_new_user = (is_new_user_str == 'true')

    if not all([email, name, username, role]):
        flash('Data tidak lengkap untuk registrasi Google.', 'danger')
        return redirect(url_for('login')) # Atau halaman register

    try:
        with connection.cursor() as cursor:
            # Jika memang user baru (dari modal) dan belum ada di DB
            if is_new_user:
                cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                existing_user = cursor.fetchone()

                if existing_user: # Double check, kalau ada, langsung login aja
                    flash('User sudah terdaftar, langsung login.', 'warning')
                    user_id = existing_user['id']
                    user_role = existing_user['role']
                    user_pricing = existing_user['pricing']
                else:
                    # Insert user baru dengan role yang dipilih dari modal
                    sql = "INSERT INTO users (nama, email, username, role) VALUES (%s, %s, %s, %s)"
                    cursor.execute(sql, (name, email, username, role))
                    connection.commit()
                    user_id = cursor.lastrowid
                    user_role = role
                    user_pricing = None # Default pricing for new users

                # Login user setelah registrasi (atau jika sudah ada)
                session.clear()
                session['user_id'] = user_id
                session['nama'] = name
                session['email'] = email
                session['username'] = username
                session['role'] = user_role
                session['pricing'] = user_pricing
                flash('Registrasi Google berhasil!', 'success')
                return redirect(url_for('index'))
            else:
                # Ini seharusnya tidak terjadi jika modal hanya untuk user baru
                # Tapi sebagai fallback, jika entah bagaimana user lama lewat sini
                flash('Sesi Google tidak valid atau user sudah terdaftar.', 'warning')
                return redirect(url_for('login')) # Arahkan ke login biasa

    except Exception as e:
        flash(f'Error saat memproses role Google: {e}', 'danger')
        print(f"Error processing Google role: {e}")
        return redirect(url_for('login'))
    
@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/admin-dashboard")
def dashboard():
    return render_template("admin/dashboard.html")


@app.route("/req-register", methods=["POST"])
def req_register():
    try:
        nama = request.form['nama']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        nama_perusahaan = request.form.get('nama_perusahaan')  # optional field

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)

        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email = %s OR username = %s", (email, username))
            existing_user = cursor.fetchone()

            if existing_user:
                flash('Email atau Username sudah terdaftar!', 'danger')
                return redirect(url_for('register'))

            sql = """INSERT INTO users 
                     (nama, username, email, password, role, nama_perusahaan) 
                     VALUES (%s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (nama, username, email, hashed_password, role, nama_perusahaan))
            connection.commit()

        flash('Registrasi berhasil! Silakan Login.', 'success')
        return redirect(url_for('login'))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("Error saat registrasi:", error_details)
        flash(f'Registrasi Gagal: {e}', 'danger')
        return redirect(url_for('register'))

def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'email' not in session:
                flash('Anda harus login untuk mengakses halaman ini.', 'danger')
                return redirect(url_for('login'))
            elif role and session.get('role') != role:
                flash('Anda tidak memiliki akses ke halaman ini.', 'danger')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route("/logout")
def logout():
    session.clear()  # Menghapus seluruh session
    flash("Anda telah berhasil logout.", "success")
    return redirect(url_for('login'))


@app.route('/') # Change this from '/dashboard' if dashboard is your main page
@app.route('/dashboard') # Keep this if you also want /dashboard to work
def index():
    print("Role:", session.get('role'))
     # <-- RENAME the function from 'dashboard' to 'index'
    if 'user_id' not in session:
        flash("Please login to access this your dashboard.", "danger")
        return redirect(url_for('login'))

    user_id = session['user_id']
    history_data = []
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT
                    created_at, job_description,
                    best_cv_path, similarity_score,
                    best_cv_path2, similarity_score2,
                    best_cv_path3, similarity_score3
                FROM history
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 5
            """
            cursor.execute(sql, (user_id,))
            results = cursor.fetchall()

            if not results:
                print("DEBUG: No history results found for this user.")

            for row in results:
                candidates_for_this_scan = []

                if row['best_cv_path']:
                    candidates_for_this_scan.append({
                        'name': os.path.basename(row['best_cv_path']),
                        'score': row['similarity_score']
                    })
                if row['best_cv_path2']:
                    candidates_for_this_scan.append({
                        'name': os.path.basename(row['best_cv_path2']),
                        'score': row['similarity_score2']
                    })
                if row['best_cv_path3']:
                    candidates_for_this_scan.append({
                        'name': os.path.basename(row['best_cv_path3']),
                        'score': row['similarity_score3']
                    })

                candidates_for_this_scan.sort(key=lambda x: x['score'], reverse=True)

                history_data.append({
                    'created_at': row['created_at'].strftime('%d %B %Y %H:%M'),
                    'job_description': row['job_description'],
                    'candidates': candidates_for_this_scan
                })

    except Exception as e:
        flash(f"Gagal memuat history scan: {e}", "danger")
        print(f"CRITICAL ERROR loading history: {e}")
        import traceback
        traceback.print_exc()

    return render_template('index.html', history=history_data) 

# ADMIN - Users
@app.route("/admin/add-user")
def add_user_page():
    return render_template("admin/add-user.html")

@app.route('/admin/users')
def users():
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM users"
            cursor.execute(sql)
            result = cursor.fetchall()  # result diharapkan menjadi list of dictionaries
            return render_template("admin/users.html", users=result)  # Kirim data 'users' ke template
    except Exception as e:
        return render_template("admin/users.html", users=[], error=str(e))  # Kirim error ke template jika gagal

@app.route('/admin/create-user', methods=['POST'])
def create_user():
    try:
        nama = request.form['nama']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']

        # Hash password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)

        # Simpan ke database
        with connection.cursor() as cursor:
            sql = "INSERT INTO users (nama, username, email, password, role) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(sql, (nama, username, email, hashed_password, role))
            connection.commit()

        flash('User berhasil ditambahkan!', 'success')
        return redirect(url_for('users'))
    except Exception as e:
        flash(f'Gagal menambahkan user: {e}', 'danger')
        return redirect(url_for('users'))

@app.route('/admin/update-user', methods=['POST'])
def update_user():
    try:
        user_id = request.form['user_id']
        nama = request.form['nama']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Pengondisian Password - Jika password diisi, hash password baru
        if password:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)
            update_sql = "UPDATE users SET nama = %s, username = %s, email = %s, password = %s WHERE id = %s"
            data = (nama, username, email, hashed_password, user_id)
        else:
            update_sql = "UPDATE users SET nama = %s, username = %s, email = %s WHERE id = %s"
            data = (nama, username, email, user_id)

        # Eksekusi SQL untuk update user
        with connection.cursor() as cursor:
            cursor.execute(update_sql, data)
            connection.commit()

        flash('User berhasil diperbarui!', 'success')
        return redirect(url_for('users'))
    except Exception as e:
        flash(f'Gagal memperbarui user: {e}', 'danger')
        return redirect(url_for('users'))

@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    try:
        # Eksekusi query untuk menghapus user berdasarkan ID
        with connection.cursor() as cursor:
            sql = "DELETE FROM users WHERE id = %s"
            cursor.execute(sql, (user_id,))
            connection.commit()

        flash('User berhasil dihapus!', 'success')
    except Exception as e:
        flash(f'Gagal menghapus user: {e}', 'danger')
    return redirect(url_for('users'))

# ADMIN - Messages
@app.route('/admin/messages')
def pesan():
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM messages"
            cursor.execute(sql)
            result = cursor.fetchall() 
            return render_template("admin/message.html", pesan=result) 
    except Exception as e:
        return render_template("admin/message.html", pesan=[], error=str(e))

@app.route('/admin/delete-msg/<int:msg_id>', methods=['POST'])
def delete_msg(msg_id):
    try:
        with connection.cursor() as cursor:
            sql = "DELETE FROM messages WHERE id = %s"
            cursor.execute(sql, (msg_id,))
            connection.commit()

        flash('Message berhasil dihapus!', 'success')
    except Exception as e:
        flash(f'Gagal menghapus message: {e}', 'danger')
    return redirect(url_for('pesan'))

# ADMIN - Scan History
@app.route('/admin/scan-history')
def scan_history():
    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT history.*, users.username, users.nama, users.email 
            FROM history
            JOIN users ON history.user_id = users.id
            ORDER BY history.created_at DESC
            """
            cursor.execute(sql)
            result = cursor.fetchall() 
            pprint(result)
            return render_template("admin/scan_history.html", scan_history=result) 
    except Exception as e:
        pprint(e)
        return render_template("admin/scan_history.html", scan_history=[], error=str(e))
    
# ADMIN - Pricing Transaction
@app.route('/admin/pricing_transaction')
def pricing_transaction():
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT pricing.*, users.nama
                FROM pricing
                JOIN users ON pricing.user_id = users.id
            """
            cursor.execute(sql)
            result = cursor.fetchall()
            return render_template("admin/pricing_transaction.html", pricing_transaction=result)
    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "danger")
        return render_template("admin/pricing_transaction.html", pricing_transaction=[])

@app.route('/admin/update_status_transaction/<int:transaction_id>', methods=['POST'])
def update_status_transaction(transaction_id):
    status = request.form.get('status')
    
    # Validasi status
    if status not in ['pending', 'paid']:
        flash("Status tidak valid", "danger")
        return redirect(url_for('pricing_transaction'))
    
    try:
        with connection.cursor() as cursor:
            # Ambil user_id dari transaksi untuk memperbarui tabel users
            sql_get_user_id = """
                SELECT user_id
                FROM pricing
                WHERE id = %s
            """
            cursor.execute(sql_get_user_id, (transaction_id,))
            result = cursor.fetchone()
            
            if not result:
                flash("Transaksi tidak ditemukan", "danger")
                return redirect(url_for('pricing_transaction'))
            
            user_id = result['user_id']

            # Update status transaksi di tabel pricing
            sql_update_pricing = """
                UPDATE pricing
                SET status = %s
                WHERE id = %s
            """
            cursor.execute(sql_update_pricing, (status, transaction_id))
            
            # Update kolom pricing di tabel users berdasarkan status
            if status == 'paid':
                sql_update_users = """
                    UPDATE users
                    SET pricing = 'plus'
                    WHERE id = %s
                """
            elif status == 'pending':
                sql_update_users = """
                    UPDATE users
                    SET pricing = NULL
                    WHERE id = %s
                """
            cursor.execute(sql_update_users, (user_id,))

            # Commit perubahan
            connection.commit()
            
        flash("Status berhasil diperbarui dan data pengguna diperbarui", "success")
        return redirect(url_for('pricing_transaction'))
    
    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "danger")
        return redirect(url_for('pricing_transaction'))

  
@app.route('/admin/delete-scan/<int:scan_id>', methods=['POST'])
def delete_scan(scan_id):
    try:
        with connection.cursor() as cursor:
            sql = "DELETE FROM history WHERE id = %s"
            cursor.execute(sql, (scan_id,))
            connection.commit()

        flash('Scan History berhasil dihapus!', 'success')
    except Exception as e:
        flash(f'Gagal menghapus scan history: {e}', 'danger')
    return redirect(url_for('scan_history'))

@app.route('/admin/delete-transaction/<int:transaction_id>', methods=['POST'])
def delete_transaction(transaction_id):
    try:
        with connection.cursor() as cursor:
            sql = "DELETE FROM pricing WHERE id = %s"
            cursor.execute(sql, (transaction_id,))
            connection.commit()

        flash('Pricing berhasil dihapus!', 'success')
    except Exception as e:
        flash(f'Gagal menghapus pricing: {e}', 'danger')
    return redirect(url_for('pricing_transaction'))

@app.route('/admin/detail-scan/<int:scan_id>', methods=['GET'])
def detail_scan(scan_id):
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM history WHERE id = %s"
            cursor.execute(sql, (scan_id))
            result = cursor.fetchone()

            # Jika data tidak ditemukan
            if not result:
                flash(f"Detail scan dengan ID {scan_id} tidak ditemukan!", "error")
                return redirect(url_for('scan_history'))

            # Kirim data ke template
            return render_template("admin/detail_scan.html", history=result)

    except Exception as e:
        # Tangani error dengan pesan dan arahkan kembali
        flash(f"Terjadi kesalahan: {str(e)}", "error")
        return redirect(url_for('scan_history'))
    
# ADMIN - Feedback History
@app.route('/admin/feedback-history')
def feedback_history():
    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT feedback.*, users.username, users.nama, users.email 
            FROM feedback
            JOIN users ON feedback.user_id = users.id
            ORDER BY feedback.created_at DESC
            """
            cursor.execute(sql)
            result = cursor.fetchall() 
            pprint(result)
            return render_template("admin/feedback_history.html", feedback_history=result) 
    except Exception as e:
        pprint(e)
        return render_template("admin/feedback_history.html", feedback_history=[], error=str(e))

@app.route('/admin/delete-feedback/<int:feedback_id>', methods=['POST'])
def delete_feedback(feedback_id):
    try:
        with connection.cursor() as cursor:
            sql = "DELETE FROM feedback WHERE id = %s"
            cursor.execute(sql, (feedback_id,))
            connection.commit()

        flash('Feedback History berhasil dihapus!', 'success')
    except Exception as e:
        flash(f'Gagal menghapus feedback history: {e}', 'danger')
    return redirect(url_for('feedback_history'))

@app.route('/admin/detail-feedback/<int:feedback_id>', methods=['GET'])
def detail_feedback(feedback_id):
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM feedback WHERE id = %s"
            cursor.execute(sql, (feedback_id,))
            result = cursor.fetchone()

            # Parse kolom feedback yang berupa string JSON
            if result['feedback']:
                result['feedback'] = json.loads(result['feedback'])
            else:
                result['feedback'] = []

            # Jika data tidak ditemukan
            if not result:
                flash(f"Detail feedback dengan ID {feedback_id} tidak ditemukan!", "error")
                return redirect(url_for('feedback_history'))

            # Kirim data ke template
            return render_template("admin/detail_feedback.html", history=result)

    except Exception as e:
        # Tangani error dengan pesan dan arahkan kembali
        flash(f"Terjadi kesalahan: {str(e)}", "error")
        return redirect(url_for('feedback_history'))

#opsi card CEKCV & CEKFEEDBACK
@app.route("/option")
def option():
    # Pastikan user sudah login untuk mengakses halaman ini
    if 'user_id' not in session:
        flash("Please login to access this feature", "danger")
        return redirect(url_for('login'))

    user_role = session.get('role') # Ambil role dari sesi

    # Kirim role ke template
    return render_template("option.html", user_role=user_role)

if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)