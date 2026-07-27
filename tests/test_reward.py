from __future__ import annotations
import unittest
import re
import string

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
        
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
        
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(final_answer: str, ground_truth_answer: str, cited_ids: list[str], gt_citations: list[str]):
    ans_clean = normalize_answer(final_answer) if final_answer else ""
    gt_clean = normalize_answer(ground_truth_answer) if ground_truth_answer else ""
    
    # Strict standardized exact match (EM)
    exact_match = 1.0 if ans_clean == gt_clean else 0.0
    
    # Substring match (contains)
    contains = 1.0 if gt_clean in ans_clean else 0.0
    
    # Token F1
    token_f1 = compute_token_f1(final_answer, ground_truth_answer)

    # Citation metrics
    cited_set = set(cited_ids)
    gt_set = set(gt_citations)
    
    intersection = cited_set & gt_set
    precision = len(intersection) / len(cited_set) if cited_set else 0.0
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    citation_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "exact_match": exact_match,
        "contains": contains,
        "token_f1": token_f1,
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": citation_f1
    }

class TestRewardCalculation(unittest.TestCase):
    def test_metrics_calculation(self):
        # Case 1: Strict exact match and perfect citations
        m1 = compute_metrics(
            final_answer="Canada",
            ground_truth_answer="Canada",
            cited_ids=["maple_leaf_flag"],
            gt_citations=["maple_leaf_flag"]
        )
        self.assertEqual(m1["exact_match"], 1.0)
        self.assertEqual(m1["contains"], 1.0)
        self.assertEqual(m1["token_f1"], 1.0)
        self.assertEqual(m1["citation_f1"], 1.0)

        # Case 2: Contains but not exact match (with punctuation and prefix)
        m2 = compute_metrics(
            final_answer="The answer is Paris.",
            ground_truth_answer="Paris",
            cited_ids=["france_capital"],
            gt_citations=["france_capital"]
        )
        self.assertEqual(m2["exact_match"], 0.0)
        self.assertEqual(m2["contains"], 1.0)
        # "answer is paris" (3 tokens) vs "paris" (1 token). Common = {"paris"}. F1 = 2 * (1/3) * 1 / (4/3) = 0.5
        self.assertAlmostEqual(m2["token_f1"], 0.5)

        # Case 3: Truncated water check
        m3 = compute_metrics(
            final_answer="The answer is 1945.",
            ground_truth_answer="1945",
            cited_ids=[],
            gt_citations=["ww2_end_year"]
        )
        self.assertEqual(m3["exact_match"], 0.0)
        self.assertEqual(m3["contains"], 1.0)
        self.assertAlmostEqual(m3["token_f1"], 0.5)

if __name__ == "__main__":
    unittest.main()
