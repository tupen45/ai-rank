from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and prepare model once
df = pd.read_csv('wbjee_mca_cutoff.csv')
df.columns = df.columns.str.strip()
df = df.rename(columns={
    'Closing Rank': 'Rank',
    'Institute': 'College',
    'Category ': 'Category'
})

df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
df = df.dropna(subset=['Rank', 'Category', 'College'])

category_encoder = LabelEncoder()
college_encoder = LabelEncoder()
df['Category_enc'] = category_encoder.fit_transform(df['Category'])
df['College_enc'] = college_encoder.fit_transform(df['College'])

X = df[['Rank', 'Category_enc']]
y = df['College_enc']

model = RandomForestClassifier()
model.fit(X, y)

@app.route('/predict', methods=['GET'])

def predict():
    rank = int(request.args.get('rank', 0))
    category = request.args.get('category', '')

    try:
        category_enc = category_encoder.transform([category])[0]
        pred_enc = model.predict([[rank, category_enc]])[0]
        college = college_encoder.inverse_transform([pred_enc])[0]
        return jsonify({'college': college})
    except:
        return jsonify({'error': 'Invalid category'}), 400

if __name__ == '__main__':
    app.run(debug=True)
