import os

import spacy
from flask import Flask, render_template, jsonify, request

from src.components import QueryProcessor, DocumentRetrieval, PassageRetrieval, AnswerExtractor

app = Flask(__name__)
SPACY_MODEL = os.environ.get('SPACY_MODEL', 'en_core_web_sm')
QA_MODEL = os.environ.get('QA_MODEL', 'distilbert-base-cased-distilled-squad')
nlp = spacy.load(SPACY_MODEL, disable=['ner', 'parser', 'textcat'])
query_processor = QueryProcessor(nlp)
document_retriever = DocumentRetrieval()
passage_retriever = PassageRetrieval(nlp)
answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answer-question', methods=['POST'])
def analyzer():
    data = request.get_json()
    question = data.get('question')

    query = query_processor.generate_query(question)
    docs = document_retriever.search(query)
    passage_retriever.fit(docs)
    passages = passage_retriever.most_similar(question)
    answers = answer_extractor.extract(question, passages)
    return jsonify(answers)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
