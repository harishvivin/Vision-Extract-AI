from pathlib import Path
from src.pipeline import ExtractionPipeline
from src.qa_engine import DocumentQAEngine
from config import OUTPUTS_DIR, LOGS_DIR
import traceback

pdf = Path('INPUT_images_and_questions.pdf')
print('PDF exists:', pdf.exists())
try:
    pipeline = ExtractionPipeline(output_dir=OUTPUTS_DIR, log_dir=LOGS_DIR)
    print('Pipeline created')
    results = pipeline.run(pdf)
    print('Pipeline run completed. Pages:', len(results))
    qa = DocumentQAEngine(outputs_dir=OUTPUTS_DIR)
    print('Created QA engine')
    sid = qa.purge_and_create_session(pdf, pdf.name)
    print('Session created', sid)
except Exception as e:
    print('Exception during processing:')
    traceback.print_exc()
