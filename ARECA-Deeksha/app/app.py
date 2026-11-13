import os
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'areca_secret_key_for_session'

# Configuration for file uploads
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username and password:
            session['user'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Please enter valid credentials.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['user'])


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'fruit_image' not in request.files:
        flash('No file selected!', 'warning')
        return redirect(url_for('index'))

    file = request.files['fruit_image']

    if file.filename == '':
        flash('No image selected for upload!', 'warning')
        return redirect(url_for('index'))

    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Mock ML Classification Result (Replace with actual model later)
        mock_result = {
            "disease_name": "Mahali / Koleroga",
            "scientific_name": "Phytophthora palmivora",
            "confidence": "96.4%",
            "uploaded_image": f"uploads/{file.filename}",
            "heatmap_image": f"uploads/{file.filename}",
            "advisory": "Apply 1% Bordeaux mixture on the tender nuts and bunches preventively before heavy monsoon rains."
        }

        return render_template('result.html', result=mock_result)


@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('history.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)