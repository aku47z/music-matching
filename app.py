"""
MIDI Plagiarism Detector - Web Interface
Flask backend for the web-based plagiarism detector.
"""

import os
import tempfile
import base64
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Local imports
from feature_extractor import extract_features_from_midi, features_to_tuples
from ngram_similarity import compute_baseline_similarity
from bipartite_matcher import (
    compute_plagiarism_score, 
)
from visualizer import visualize_bipartite_graph

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')

ALLOWED_EXTENSIONS = {'mid', 'midi'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_sample_files():
    """Get list of sample MIDI files."""
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for f in sorted(os.listdir(SAMPLES_DIR)):
            if allowed_file(f):
                samples.append({
                    'name': f.replace('.mid', '').replace('_', ' ').replace('-', ' '),
                    'filename': f
                })
    return samples


def analyze_files(path_a, path_b, max_nodes_display=15):
    """Run plagiarism analysis on two MIDI files.
    
    Args:
        path_a: Path to first MIDI file
        path_b: Path to second MIDI file
        max_nodes_display: Number of top nodes to display (None for all)
    """
    # Extract features
    features_a = extract_features_from_midi(path_a)
    features_b = extract_features_from_midi(path_b)
    
    if len(features_a) < 2 or len(features_b) < 2:
        return {
            'error': 'One or both files have insufficient notes for analysis.',
            'success': False
        }
    
    # Convert to tuples
    tuples_a = features_to_tuples(features_a)
    tuples_b = features_to_tuples(features_b)
    
    # N-Gram baseline
    ngram_score = compute_baseline_similarity(
        tuples_a, tuples_b, 
        n=3, 
        use_quantization=True
    )
    
    # Bipartite matching analysis
    result = compute_plagiarism_score(tuples_a, tuples_b, tempo_bpm=120)
    
    hook_score = result['hook_score']
    
    # Top matches
    top_matches = []
    loc_matches = sorted(result.get('localized_matches', []), key=lambda x: x.weight, reverse=True)[:5]
    for m in loc_matches:
        top_matches.append({
            'score': round(m.weight, 4),
            'time_a': m.time_a,
            'time_b': m.time_b
        })
    
    # Generate bipartite graph visualization
    graph_image = None
    try:
        if result['graph'].number_of_nodes() > 0:
            fig = visualize_bipartite_graph(
                result['graph'],
                result['weight_matrix'],
                result.get('top_k_matches', []),
                title='Bipartite Graph - Top Matching Fragments',
                figsize=(12, 8),
                max_nodes_display=max_nodes_display
            )
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            graph_image = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
    except Exception as e:
        pass  # Silently handle graph generation errors
    
    return {
        'success': True,
        'ngram_score': round(ngram_score, 4),
        'hook_score': round(hook_score, 4),
        'fragments_a': len(result['fragments_a']),
        'fragments_b': len(result['fragments_b']),
        'notes_a': len(features_a),
        'notes_b': len(features_b),
        'top_matches': top_matches,
        'graph_image': graph_image
    }


@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/samples')
def api_samples():
    """Get list of sample files."""
    return jsonify(get_sample_files())


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Analyze two MIDI files for plagiarism."""
    try:
        # Get max_nodes parameter
        max_nodes_param = request.form.get('max_nodes', '15')
        max_nodes = None if max_nodes_param == 'all' else int(max_nodes_param)
        
        path_a = None
        path_b = None
        temp_files = []
        
        # Handle Song A
        if 'file_a' in request.files and request.files['file_a'].filename:
            file_a = request.files['file_a']
            if allowed_file(file_a.filename):
                filename = secure_filename(file_a.filename)
                path_a = os.path.join(app.config['UPLOAD_FOLDER'], f"upload_a_{filename}")
                file_a.save(path_a)
                temp_files.append(path_a)
        elif request.form.get('sample_a'):
            # Don't use secure_filename for server samples - they're trusted files
            sample_name = request.form.get('sample_a')
            path_a = os.path.join(SAMPLES_DIR, sample_name)
        
        # Handle Song B
        if 'file_b' in request.files and request.files['file_b'].filename:
            file_b = request.files['file_b']
            if allowed_file(file_b.filename):
                filename = secure_filename(file_b.filename)
                path_b = os.path.join(app.config['UPLOAD_FOLDER'], f"upload_b_{filename}")
                file_b.save(path_b)
                temp_files.append(path_b)
        elif request.form.get('sample_b'):
            # Don't use secure_filename for server samples - they're trusted files
            sample_name = request.form.get('sample_b')
            path_b = os.path.join(SAMPLES_DIR, sample_name)
        
        # Validate
        if not path_a or not path_b:
            return jsonify({'error': 'Please provide two MIDI files to compare.', 'success': False})
        
        if not os.path.exists(path_a):
            return jsonify({'error': f'File A not found: {path_a}', 'success': False})
        
        if not os.path.exists(path_b):
            return jsonify({'error': f'File B not found: {path_b}', 'success': False})
        
        # Run analysis
        result = analyze_files(path_a, path_b, max_nodes_display=max_nodes)
        
        # Cleanup temp files
        for f in temp_files:
            try:
                os.remove(f)
            except:
                pass
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False})

def get_forensic_scores(path_a, path_b):
    """Get both overall plagiarism score and max segment score between two songs."""
    try:
        features_a = extract_features_from_midi(path_a)
        features_b = extract_features_from_midi(path_b)
        
        if len(features_a) < 2 or len(features_b) < 2:
            return 0.0, 0.0
        
        tuples_a = features_to_tuples(features_a)
        tuples_b = features_to_tuples(features_b)
        
        result = compute_plagiarism_score(tuples_a, tuples_b, tempo_bpm=120)
        
        # Overall combined score
        overall_score = result.get('combined_score', 0.0)
        
        # Get the maximum segment match score
        loc_matches = result.get('localized_matches', [])
        if loc_matches:
            max_segment_score = max(m.weight for m in loc_matches)
        else:
            max_segment_score = result.get('hook_score', 0.0)
        
        return overall_score, max_segment_score
    except Exception:
        return 0.0, 0.0


@app.route('/api/forensic', methods=['POST'])
def api_forensic():
    """Run forensic ranking mode - compare query against all songs."""
    try:
        path_a = None
        temp_files = []
        
        # Handle Song A (Query Melody)
        if 'file_a' in request.files and request.files['file_a'].filename:
            file_a = request.files['file_a']
            if allowed_file(file_a.filename):
                filename = secure_filename(file_a.filename)
                path_a = os.path.join(app.config['UPLOAD_FOLDER'], f"forensic_a_{filename}")
                file_a.save(path_a)
                temp_files.append(path_a)
        elif request.form.get('sample_a'):
            sample_name = request.form.get('sample_a')
            path_a = os.path.join(SAMPLES_DIR, sample_name)
        
        if not path_a or not os.path.exists(path_a):
            return jsonify({'error': 'Query melody (Song A) not found.', 'success': False})
        
        # Handle Song B (Suspect Song)
        suspect_path = None
        if 'file_b' in request.files and request.files['file_b'].filename:
            file_b = request.files['file_b']
            if allowed_file(file_b.filename):
                filename = secure_filename(file_b.filename)
                suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], f"forensic_b_{filename}")
                file_b.save(suspect_path)
                temp_files.append(suspect_path)
        elif request.form.get('sample_b'):
            sample_name = request.form.get('sample_b')
            suspect_path = os.path.join(SAMPLES_DIR, sample_name)
        
        if not suspect_path or not os.path.exists(suspect_path):
            return jsonify({'error': 'Suspect song (Song B) not found.', 'success': False})
        
        # Build comparison library: ALL sample songs
        all_samples = get_sample_files()
        
        # Build library with all songs, marking the suspect
        library = []
        suspect_in_samples = False
        
        for sample in all_samples:
            sample_path = os.path.join(SAMPLES_DIR, sample['filename'])
            # Skip Song A itself
            if sample['filename'] == os.path.basename(path_a):
                continue
            # Check if this is the suspect song
            is_suspect = (sample['filename'] == os.path.basename(suspect_path))
            if is_suspect:
                suspect_in_samples = True
            library.append({
                'name': sample['filename'],
                'path': sample_path,
                'is_suspect': is_suspect
            })
        
        # If suspect is an uploaded file (not in samples), add it
        if not suspect_in_samples and suspect_path:
            library.insert(0, {
                'name': os.path.basename(suspect_path),
                'path': suspect_path,
                'is_suspect': True
            })
        
        # Run batch analysis against all songs
        results = []
        total = len(library)
        for idx, item in enumerate(library):
            overall_score, segment_score = get_forensic_scores(path_a, item['path'])
            results.append({
                'name': item['name'].replace('.mid', '').replace('_', ' '),
                'overall_score': float(overall_score),
                'segment_score': float(segment_score),
                'is_suspect': bool(item['is_suspect'])
            })
        
        # Sort by overall plagiarism score (descending) - default sorting
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Find suspect rank (based on overall score)
        suspect_rank = next((i for i, r in enumerate(results) if r['is_suspect']), -1)
        suspect_is_top = bool(suspect_rank == 0)
        
        # Cleanup temp files
        for f in temp_files:
            try:
                os.remove(f)
            except:
                pass
        
        return jsonify({
            'success': True,
            'rankings': results,
            'total_compared': len(results),
            'suspect_rank': suspect_rank + 1 if suspect_rank >= 0 else -1,
            'suspect_is_top': suspect_is_top
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MIDI Plagiarism Detector - Web Interface")
    print("="*60)
    print(f"\nSamples directory: {SAMPLES_DIR}")
    print(f"Sample files found: {len(get_sample_files())}")
    print("\nStarting server at http://0.0.0.0:5001")
    print("Access via: http://localhost:5001 or http://127.0.0.1:5001")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', debug=True, port=5001)
