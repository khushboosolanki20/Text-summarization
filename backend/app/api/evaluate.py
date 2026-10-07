"""POST /api/evaluate: score any summary against a reference with ROUGE."""

from fastapi import APIRouter

from app.api.schemas import EvaluateRequest, EvaluateResponse
from app.errors import EmptyDocumentError
from app.evaluation.faithfulness import check_faithfulness
from app.evaluation.rouge import compute_rouge
from app.preprocessing.cleaner import count_words
from app.preprocessing.pipeline import preprocess
from app.summarizers.base import compression_ratio

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluateResponse, summary="Evaluate a summary with ROUGE")
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """
    Compute ROUGE-1/2/L/Lsum between `summary` and `reference_summary`, e.g.
    to compare an IntelliSum summary with a summary written by hand. If
    `original_text` is given, the compression ratio and the experimental
    faithfulness check are also returned.
    """
    if not request.summary.strip():
        raise EmptyDocumentError("The summary to evaluate is empty.")
    rouge = compute_rouge(request.summary, request.reference_summary)  # rejects an empty reference

    ratio = faithfulness = None
    if request.original_text and request.original_text.strip():
        source = preprocess(request.original_text)
        ratio = compression_ratio(source.word_count, count_words(request.summary))
        faithfulness = check_faithfulness(request.summary, source.sentences)

    return EvaluateResponse(
        rouge1=rouge["rouge1"]["f1"],
        rouge2=rouge["rouge2"]["f1"],
        rougeL=rouge["rougeL"]["f1"],
        rouge=rouge,
        summary_word_count=count_words(request.summary),
        reference_word_count=count_words(request.reference_summary),
        compression_ratio=ratio,
        faithfulness=faithfulness,
    )
