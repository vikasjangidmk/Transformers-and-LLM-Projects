from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load NLP pipelines
sentiment_pipeline = pipeline('sentiment-analysis')
generation_pipeline = pipeline('text-generation')
translation_pipeline = pipeline('translation', model="Helsinki-NLP/opus-mt-fr-en")
summarization_pipeline = pipeline('summarization')
NER_pipeline = pipeline("ner", grouped_entities=True)

@app.route('/', methods=['POST', 'GET'])
def index():
    sent = gen = trans = ner = summ = None
    
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        task = request.form.get('task')

        if task == 'sentiment':
            sent = sentiment_pipeline(user_input)
        elif task == 'generation':
            gen = generation_pipeline(user_input)
        elif task == 'translation':
            trans = translation_pipeline(user_input)
        elif task == 'summarization':
            summ = summarization_pipeline(user_input)
        elif task == 'named_entity_recognition':
            ner = NER_pipeline(user_input)

    # Always return a response
    return render_template('templates.html', sent=sent, gen=gen, trans=trans, ner=ner, summ=summ)

if __name__ == "__main__":
    app.run(debug=True)
