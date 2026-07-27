from __future__ import annotations
import json
from typing import Dict, Any

ANSWERS_MAP = {
    "toy_task_001": ("Canada", "maple_leaf_flag"),
    "toy_task_002": ("William Shakespeare", "romeo_juliet_author"),
    "toy_task_003": ("Paris", "france_capital"),
    "toy_task_004": ("Mars", "mars_red_planet"),
    "toy_task_005": ("Albert Einstein", "general_relativity_creator"),
    "toy_task_006": ("H2O", "water_chemical_symbol"),
    "toy_task_007": ("Mount Everest", "tallest_mountain"),
    "toy_task_008": ("Pacific Ocean", "largest_ocean"),
    "toy_task_009": ("Leonardo da Vinci", "mona_lisa_painter"),
    "toy_task_010": ("1945", "ww2_end_year")
}

class MockLLMPolicy:
    def decide(self, obs: Any) -> str:
        tid = obs.task_id
        
        # 1. Scenario 5: Parser invalid error injection
        if tid == "toy_task_005" and obs.invalid_action_count == 0:
            return "This response is completely broken. It does not contain action or JSON tags, simulating a parsing error."

        # 2. Scenario 6: Max steps / no progress simulation
        if tid == "toy_task_006":
            return (
                f"<reasoning>I will keep searching endlessly for task {tid}</reasoning>\n"
                f"<action>\n"
                f"{{\n"
                f"  \"tool\": \"SEARCH\",\n"
                f"  \"params\": {{\"query\": \"useless query {obs.remaining_steps}\", \"topk\": 3}}\n"
                f"}}\n"
                f"</action>"
            )

        # 3. If remaining steps are critical, force ANSWER
        if obs.remaining_steps <= 1:
            return self._make_answer_text(obs)

        # 4. Scenario 2: SEARCH returns nothing
        if tid == "toy_task_002":
            if "SEARCH" not in obs.trajectory:
                return (
                    f"<reasoning>I need to search for Romeo and Juliet author using a gibberish query.</reasoning>\n"
                    f"<action>\n"
                    f"{{\n"
                    f"  \"tool\": \"SEARCH\",\n"
                    f"  \"params\": {{\"query\": \"gibberish_query_returning_nothing_123\", \"topk\": 3}}\n"
                    f"}}\n"
                    f"</action>"
                )
            else:
                return (
                    f"<reasoning>The search returned no candidate chunks. I cannot find the answer.</reasoning>\n"
                    f"<action>\n"
                    f"{{\n"
                    f"  \"tool\": \"ANSWER\",\n"
                    f"  \"params\": {{\"answer_text\": \"No info found.\", \"cited_chunk_ids\": []}}\n"
                    f"}}\n"
                    f"</action>"
                )

        # 5. Check if SEARCH has run
        if "SEARCH" not in obs.trajectory:
            return (
                f"<reasoning>I will search for the user query: '{obs.user_query}'</reasoning>\n"
                f"<action>\n"
                f"{{\n"
                f"  \"tool\": \"SEARCH\",\n"
                f"  \"params\": {{\"query\": \"{obs.user_query}\", \"topk\": 3}}\n"
                f"}}\n"
                f"</action>"
            )

        # 6. Check if READ has run
        if "READ" not in obs.trajectory:
            # Scenario 3: Read invalid chunk
            if tid == "toy_task_003":
                return (
                    f"<reasoning>I will try to read an invalid chunk ID first to test error handling.</reasoning>\n"
                    f"<action>\n"
                    f"{{\n"
                    f"  \"tool\": \"READ\",\n"
                    f"  \"params\": {{\"chunk_ids\": [\"non_existent_chunk_id\"]}}\n"
                    f"}}\n"
                    f"</action>"
                )
            
            # Normal read: read the correct chunk ID from the map
            _, correct_chunk_id = ANSWERS_MAP.get(tid, ("Unknown", ""))
            return (
                f"<reasoning>Search returned candidate chunks. Let's read the correct chunk: {correct_chunk_id}</reasoning>\n"
                f"<action>\n"
                f"{{\n"
                f"  \"tool\": \"READ\",\n"
                f"  \"params\": {{\"chunk_ids\": [\"{correct_chunk_id}\"]}}\n"
                f"}}\n"
                f"</action>"
            )

        # 7. Scenario 3 recovery: If READ was run once but only read non-existent chunk_id, try reading correct one
        if tid == "toy_task_003" and len(obs.read_summaries) == 0:
            _, correct_chunk_id = ANSWERS_MAP.get(tid, ("Unknown", ""))
            return (
                f"<reasoning>My previous read failed. Let's read the correct candidate chunk: {correct_chunk_id}</reasoning>\n"
                f"<action>\n"
                f"{{\n"
                f"  \"tool\": \"READ\",\n"
                f"  \"params\": {{\"chunk_ids\": [\"{correct_chunk_id}\"]}}\n"
                f"}}\n"
                f"</action>"
            )

        # 8. Submit final ANSWER
        return self._make_answer_text(obs)

    def _make_answer_text(self, obs: Any) -> str:
        tid = obs.task_id
        
        # Scenario 4: ANSWER with no citations
        if tid == "toy_task_004":
            return (
                f"<reasoning>I found that Mars is the Red Planet, and I will answer with no citations.</reasoning>\n"
                f"<action>\n"
                f"{{\n"
                f"  \"tool\": \"ANSWER\",\n"
                f"  \"params\": {{\"answer_text\": \"Mars is the answer.\", \"cited_chunk_ids\": []}}\n"
                f"}}\n"
                f"</action>"
            )

        ans, correct_chunk_id = ANSWERS_MAP.get(tid, ("Unknown", ""))
        return (
            f"<reasoning>I have read the documents. The answer is {ans}. I will submit the final answer citing {correct_chunk_id}.</reasoning>\n"
            f"<action>\n"
            f"{{\n"
            f"  \"tool\": \"ANSWER\",\n"
            f"  \"params\": {{\"answer_text\": \"The answer is {ans}.\", \"cited_chunk_ids\": [\"{correct_chunk_id}\"]}}\n"
            f"}}\n"
            f"</action>"
        )
