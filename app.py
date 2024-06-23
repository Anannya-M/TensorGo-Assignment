from flask import Flask, request, render_template, send_file, redirect, url_for, session
import pandas as pd
import matplotlib.pyplot as plt
from transformers import LlamaTokenizer
from io import BytesIO ,StringIO
import os
import secrets
import seaborn as sns
import chardet
from llama_cpp import Llama
from transformers import  AutoTokenizer


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a random secret key

#llama_pipeline = pipeline('text-generation', model='gpt2')
model_path = "C:\\Users\\ASUS\\Downloads\\llama-2-7b-chat.Q2_K.gguf"
llm = Llama(model_path=model_path)

#print(llm("what are the large language model and how do we use them"))


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def read_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1')
    
def summarize_csv(data, num_rows=5):
    summary = data.head(num_rows).to_string()
    stats = data.describe(include='all').to_string()
    return f"Summary:\n{summary}\n\nStatistics:\n{stats}"

def chunkify_text(text, max_length=512):
    chunks = []
    current_chunk = ""
    
    for word in text.split():
        if len(current_chunk) + len(word) <= max_length:
            current_chunk += f" {word}"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = f"{word}"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_context_from_csv(data):
    return data.to_string()

def calculate_statistics(data):
    numeric_data = data.select_dtypes(include=[float, int])
    stats = {
        'mean': numeric_data.mean().to_dict(),
        'median': numeric_data.median().to_dict(),
        'mode': numeric_data.mode().iloc[0].to_dict(),
        'std_dev': numeric_data.std().to_dict(),
        'correlation': numeric_data.corr().to_dict()
    }
    return stats

def generate_plot(data, plot_type, columns):
    buf = BytesIO()
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'histogram':
        plt.hist(data[columns[0]], bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram of {columns[0]}')
        plt.xlabel(columns[0])
        plt.ylabel('Frequency')
    elif plot_type == 'scatter':
        plt.scatter(data[columns[0]], data[columns[1]], alpha=0.7, color='blue')
        plt.title(f'Scatter Plot of {columns[0]} vs {columns[1]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
    elif plot_type == 'line':
        plt.plot(data[columns[0]], color='blue')
        plt.title(f'Line Plot of {columns[0]}')
        plt.xlabel('Index')
        plt.ylabel(columns[0])
    elif plot_type == 'box':
        plt.boxplot(data[columns].dropna().values)
        plt.title(f'Box Plot of {", ".join(columns)}')
        plt.xlabel('Columns')
        plt.ylabel('Values')
        plt.xticks(range(1, len(columns)+1), columns)
    elif plot_type == 'bar':
        data[columns[0]].value_counts().plot(kind='bar', color='blue')
        plt.title(f'Bar Plot of {columns[0]}')
        plt.xlabel(columns[0])
        plt.ylabel('Counts')
    elif plot_type == 'pie':
        data[columns[0]].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'Pie Chart of {columns[0]}')
        plt.ylabel('')
    elif plot_type == 'area':
        data[columns].plot(kind='area', alpha=0.4)
        plt.title(f'Area Plot of {", ".join(columns)}')
        plt.xlabel('Index')
        plt.ylabel('Values')
    elif plot_type == 'hexbin':
        plt.hexbin(data[columns[0]], data[columns[1]], gridsize=30, cmap='Blues')
        plt.title(f'Hexbin Plot of {columns[0]} vs {columns[1]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        cb = plt.colorbar(label='Count')
    elif plot_type == 'density':
        data[columns[0]].plot(kind='density', color='blue')
        plt.title(f'Density Plot of {columns[0]}')
        plt.xlabel(columns[0])
        plt.ylabel('Density')
    elif plot_type == 'violin':
        sns.violinplot(data=data[columns].dropna())
        plt.title(f'Violin Plot of {", ".join(columns)}')
        plt.xlabel('Columns')
        plt.ylabel('Values')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf


# def answer_question(llama_pipeline, question, context):
#     input_text = f"Context: {context}\nQuestion: {question}"
#     response = llama_pipeline(input_text, max_new_tokens=200, num_return_sequences=1)
#     return response[0]['generated_text']

def answer_question(llm, question, context_chunks):
    answers = []
    for chunk in context_chunks:
        prompt = f"Context: {chunk}\n\nQuestion: {question}\n\nAnswer:"
        response = llm(prompt)
        answers.append(response['choices'][0]['text'].strip())
    return " ".join(answers)

# def answer_question(llm, question, context_chunks, max_paragraph_length=512):
#     answer_chunks = []
#     current_length = 0

#     for chunk in context_chunks:
#         prompt = f"Context: {chunk}\n\nQuestion: {question}\n\nAnswer:"
#         response = llm(prompt)
#         answer_text = response['choices'][0]['text'].strip()
        
#         # Check if adding this chunk exceeds the paragraph length
#         if current_length + len(answer_text) > max_paragraph_length:
#             break
        
#         # Append this chunk to the answer
#         answer_chunks.append(answer_text)
#         current_length += len(answer_text)
    
#     # Combine chunks into a single paragraph
#     answer = " ".join(answer_chunks)
#     return answer


@app.route('/')
def index():
    uploaded_file = session.get('uploaded_file', None)
    return render_template('index.html', uploaded_file=uploaded_file)


@app.route('/upload', methods=['POST'])
def upload_file():
    global data
    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        session['uploaded_file'] = file.filename
        data = read_csv(file_path)
        session['data'] = data.to_json()
    return redirect(url_for('index'))


@app.route('/show_data')
def show_data():
    data_json = session.get('data', None)
    if data_json:
        data = pd.read_json(data_json)
        return render_template('data.html', data=data.to_html())
    return redirect(url_for('index'))

@app.route('/dimensions')
def dimensions():
    data_json = session.get('data', None)
    if data_json:
        data = pd.read_json(data_json)
        data_info = {
            'shape': data.shape,
            'head': data.head().to_html(),
            'describe': data.describe().to_html()
        }
        return render_template('dimensions.html', data_info=data_info)
    return redirect(url_for('index'))

@app.route('/statistics', methods=['GET', 'POST'])
def statistics():
    data_json = session.get('data', None)
    if data_json:
        data = pd.read_json(data_json)
        stats = calculate_statistics(data)

        columns = data.columns.tolist()
        plot_url = None

        if request.method == 'POST':
            plot_type = request.form.get('plot_type')
            selected_columns = request.form.getlist('columns')
            if plot_type and selected_columns:
                plot_buf = generate_plot(data, plot_type, selected_columns)
                plot_url = url_for('static', filename='plot.png')
                with open('static/plot.png', 'wb') as f:
                    f.write(plot_buf.getbuffer())

        return render_template('statistics.html', stats=stats, columns=columns, plot_url=plot_url)
    return redirect(url_for('index'))

@app.route('/plot', methods=['POST'])
def plot():
    plot_type = request.form['plot_type']
    columns = request.form['columns'].split(',')
    data_json = session.get('data', None)
    if data_json:
        data = pd.read_json(data_json)
        plot_buf = generate_plot(data, plot_type, columns)
        return send_file(plot_buf, mimetype='image/png')
    return redirect(url_for('index'))


@app.route('/question', methods=['POST'])
def question():
    question = request.form['question']
    data_json = session.get('data', None)
    if data_json:
        data = pd.read_json(StringIO(data_json))
        context = summarize_csv(data)
        context_chunks = chunkify_text(context)
        answer = answer_question(llm, question, context_chunks)
        return render_template('answer.html', answer=answer)
    return redirect(url_for('index'))
 


if __name__ == '__main__':
    app.run(debug=True)
