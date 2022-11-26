from flask import Flask, request, render_template, send_from_directory
import predictor

app = Flask(__name__, template_folder='./views')


@app.route('/img/<filename>')
def img_file(filename):
    return send_from_directory('./img', filename)


@app.route('/', methods=['GET', 'POST'])
def make_prediction():
    if request.method == 'POST':
        iw_val = request.form['iw']
        if_val = request.form['if']
        vw_val = request.form['vw']
        fp_val = request.form['fp']
        prediction = predictor.predict_welding(predictor.value_scaler(iw_val, if_val, vw_val, fp_val))
        return render_template('index.html', prediction=f'Глубина (Depth) = {prediction[0][0]:.2f}; Ширина (Width) = {prediction[0][1]:.2f}')

    return render_template('index.html')


app.run('127.0.0.1', port=8200, debug=True)
