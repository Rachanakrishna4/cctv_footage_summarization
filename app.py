from flask import Flask, request, send_file, render_template_string
import summarizer

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Video Summarization App</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container py-5">
        <div class="card shadow-sm">
            <div class="card-body">
                <h2 class="card-title text-center mb-4">🎬 Video Summarization App</h2>
                <form method="post" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="video" class="form-label">Upload your video file</label>
                        <input class="form-control" type="file" name="video" id="video" required>
                    </div>
                    <div class="d-grid">
                        <button type="submit" class="btn btn-primary">Generate Summary</button>
                    </div>
                </form>
                {% if video_ready %}
                <div class="mt-4 text-center">
                    <a href="/download" class="btn btn-success">Download Summarized Video</a>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def index():
    video_ready = False
    if request.method == "POST":
        video = request.files["video"]
        video.save("input.mp4")
        summarizer.extract_frames("input.mp4")
        summarizer.cluster_frames()
        summarizer.build_video()
        video_ready = True
    return render_template_string(HTML_TEMPLATE, video_ready=video_ready)

@app.route("/download")
def download():
    return send_file("static/summary_output.mp4", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)