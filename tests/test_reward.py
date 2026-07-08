from __future__ import annotations
import unittest

def compute_metrics(final_answer: str, ground_truth_answer: str, cited_ids: list[str], gt_citations: list[str]):
    ans_clean = final_answer.lower().strip() if final_answer else ""
    gt_clean = ground_truth_answer.lower().strip() if ground_truth_answer else ""
    
    em = 1.0 if gt_clean in ans_clean else 0.0

    cited_set = set(cited_ids)
    gt_set = set(gt_citations)
    
    intersection = cited_set & gt_set
    precision = len(intersection) / len(cited_set) if cited_set else 0.0
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "em": em,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

class TestRewardCalculation(unittest.TestCase):
    def test_metrics_calculation(self):
        # Case 1: Exact match and perfect citations
        m1 = compute_metrics(
            final_answer="Canada is the answer.",
            ground_truth_answer="Canada",
            cited_ids=["maple_leaf_flag"],
            gt_citations=["maple_leaf_flag"]
        )
        self.assertEqual(m1["em"], 1.0)
        self.assertEqual(m1["precision"], 1.0)
        self.assertEqual(m1["recall"], 1.0)
        self.assertEqual(m1["f1"], 1.0)

        # Case 2: Partial citation match
        m2 = compute_metrics(
            final_answer="The answer is Canada.",
            ground_truth_answer="Canada",
            cited_ids=["maple_leaf_flag", "other_chunk"],
            gt_citations=["maple_leaf_flag"]
        )
        self.assertEqual(m2["em"], 1.0)
        self.assertEqual(m2["precision"], 0.5)
        self.assertEqual(m2["recall"], 1.0)
        self.assertAlmostEqual(m2["f1"], 0.66666667)

        # Case 3: Wrong answer, no citations
        m3 = compute_metrics(
            final_answer="France",
            ground_truth_answer="Canada",
            cited_ids=[],
            gt_citations=["maple_leaf_flag"]
        )
        self.assertEqual(m3["em"], 0.0)
        self.assertEqual(m3["precision"], 0.0)
        self.assertEqual(m3["recall"], 0.0)
        self.assertEqual(m3["f1"], 0.0)

if __name__ == "__main__":
    unittest.main()
