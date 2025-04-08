from flask import Flask, flash, render_template, request
from flask_toastr import Toastr
from werkzeug.utils import secure_filename
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import os
import magic
import psycopg2
import atexit
import config

app = Flask(__name__)
app.secret_key = "super secret key"
toastr = Toastr(app)

DB_NAME=config.db_name
DB_USER=config.db_user
DB_PASSWORD=config.db_password
DB_HOST=config.db_host
DB_PORT=config.db_port
DB_SSLMODE=config.db_sslmode
OPENAI_ENDPOINT=config.openai_endpoint
OPENAI_SUBSCRIPTION_KEY=config.openai_subscription_key
EMBEDDING_MODEL_NAME=config.embedding_model_name
FORM_RECOGNIZER_ENDPOINT=config.form_recognizer_endpoint
FORM_RECOGNIZER_KEY=config.form_recognizer_key

# Connect to Azure PostgreSQL
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    sslmode=DB_SSLMODE
)
cur = conn.cursor()

def is_pdf(file_storage):
    file_bytes = file_storage.read(2048)
    file_storage.seek(0)
    mime_type = magic.from_buffer(file_bytes, mime=True)
    return mime_type == 'application/pdf'

def chunk_text(text, chunk_size=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size
    return chunks

def store_in_db(chunks):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Create table if not exists
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id SERIAL NOT NULL PRIMARY KEY,
        chunk TEXT,
        chunk_vector VECTOR(1536)
    );
    """)
    conn.commit()

    # Insert chunks
    for chunk in chunks:
        cur.execute("INSERT INTO chunks(chunk) VALUES(%s);", (chunk,))

    conn.commit()

def vectorize_chunks():
    cur.execute("CREATE EXTENSION IF NOT EXISTS azure_ai;")
    cur.execute(f"SELECT azure_ai.set_setting('azure_openai.endpoint', '{OPENAI_ENDPOINT}');")
    cur.execute(f"SELECT azure_ai.set_setting('azure_openai.subscription_key', '{OPENAI_SUBSCRIPTION_KEY}');")
    cur.execute(f"""UPDATE chunks
                    SET chunk_vector = azure_openai.create_embeddings('{EMBEDDING_MODEL_NAME}', chunk, max_attempts => 5, retry_delay_ms => 500)
                    WHERE chunk_vector IS NULL;""")
    conn.commit()


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file_storage = request.files["pdf"]
        if not is_pdf(file_storage):
            flash("Please upload a PDF!")
        else:
            filename = secure_filename(file_storage.filename)
            file_storage.save(os.path.join("uploads", filename))
            endpoint = FORM_RECOGNIZER_ENDPOINT
            key = FORM_RECOGNIZER_KEY
            client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
            with open(f"uploads/{filename}", "rb") as f:
                poller = client.begin_analyze_document("prebuilt-layout", AnalyzeDocumentRequest(bytes_source=f.read()))
                result = poller.result()
                full_text = " ".join([" ".join(word.content for word in page.words) for page in result.pages])
            chunks = chunk_text(full_text)
            store_in_db(chunks)
            vectorize_chunks()
            flash("Uploaded the data into the database and generated embeddings!")
    return render_template("index.html")

# Cleanup function
def cleanup():
    cur.execute("DROP TABLE chunks;")
    conn.commit()
    cur.close()
    conn.close()

# Register cleanup
atexit.register(cleanup)

if __name__ == "__main__":
    app.run(debug=True)