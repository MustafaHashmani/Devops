from flask import Flask, render_template, request, jsonify
import os
from models.model import process_live_video  # Import your detection function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['OUTPUT_FOLDER'] = './static/outputs'  # For saving results

# Variable to store the last uploaded query image path
query_image_path = None

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload_samples', methods=['POST'])
def upload_samples():
    global query_image_path  # Access the global variable
    files = request.files.getlist('samples')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    
    # Save the first file as the query image
    for file in files:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Set the first uploaded file as the query image
        if query_image_path is None:
            query_image_path = file_path

    return jsonify({"message": "Uploaded Successfully!", "query_image": query_image_path})

@app.route('/process_live', methods=['POST'])
def process_live():
    global query_image_path  # Access the global query image path
    if not query_image_path:
        return jsonify({"error": "No query image available. Please upload samples first."}), 400

    # Get the image frame from the frontend (assuming base64 or file data)
    file = request.files.get('frame')
    if not file:
        return jsonify({"error": "No frame provided"}), 400
    print("Received frame:", file)  # Add this to debug the file received

    try:
        # Process the image frame (you'll need to modify your detection code)
        similarity_score = process_live_video(
            query_image_path=query_image_path,
            frame=file,
            top_n=5,
            output_dir=app.config['OUTPUT_FOLDER']
        )
        print("Similarity Score:", similarity_score)
        return jsonify({
            "message": "Processing completed",
            "results": similarity_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
