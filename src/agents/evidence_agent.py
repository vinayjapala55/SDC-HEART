"""
Agent 3: Medical Evidence and Risk Context Agent.

Responsibilities:
1. Interface with local ChromaDB collection of 2020 ACC/AHA Valvular Heart Disease Guidelines.
2. Build clinical queries based on RosaNet detected conditions.
3. Retrieve relevant evidence chunks with verifiable metadata (Section, Page, COR/LOE).
4. Provide structured clinical guideline context for detected conditions without asserting a diagnosis.
5. Identify missing clinical diagnostics required for confirmation (e.g. Echocardiogram, Doppler).
6. Return structured EvidenceAgentOutput.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from src.agents.schemas import EvidenceAgentOutput, GuidelineEvidenceChunk, DetectedCondition
from src.rag.query_builder import ClinicalQueryBuilder
from src.rag.retriever import ClinicalRetriever

DISEASE_DESCRIPTIONS = {
    "AS": "Aortic Stenosis: Narrowing of the aortic valve opening, commonly producing a harsh systolic crescendo-decrescendo murmur loudest at the right upper sternal border with radiation to the carotids.",
    "AR": "Aortic Regurgitation: Incompetence of the aortic valve causing diastolic backflow into the LV, producing a high-pitched early diastolic decrescendo murmur best heard along the left sternal border.",
    "MR": "Mitral Regurgitation: Retrograde blood flow from the LV into the LA during ventricular systole, typically characterized by a holosystolic murmur radiating to the left axilla.",
    "MS": "Mitral Stenosis: Obstruction of LV inflow across the mitral valve during diastole, classic for an opening snap followed by a low-pitched rumbling diastolic murmur loudest at the apex.",
    "N": "Normal Heart Auscultation: Physiological S1 and S2 heart sounds with crisp valve closure and absence of pathological systolic or diastolic murmurs.",
    "MULTI": "Multi-Valvular Disease: Simultaneous involvement of multiple cardiac valves, leading to complex overlapping hemodynamic interactions and mixed murmur characteristics."
}



class MedicalEvidenceAgent:
    """
    RAG-powered clinical guideline evidence retrieval and context agent.
    """

    def __init__(self, chroma_db_dir: Optional[str] = None):
        self.query_builder = ClinicalQueryBuilder()
        try:
            self.retriever = ClinicalRetriever(chroma_db_dir=chroma_db_dir)
        except Exception as e:
            print(f"[MedicalEvidenceAgent] Warning: Could not initialize ChromaDB: {e}")
            self.retriever = None

    def retrieve_clinical_evidence(
        self,
        predicted_probabilities: Dict[str, float],
        detected_conditions: List[DetectedCondition],
        session_id: str = "SESSION_UNKNOWN"
    ) -> EvidenceAgentOutput:
        """
        Retrieves targeted guideline evidence for the predicted cardiac patterns.
        """
        # Build targeted queries
        query_plan = self.query_builder.build_queries(predicted_probabilities)
        queries = query_plan.get("queries", [])

        retrieved_chunks: List[GuidelineEvidenceChunk] = [] 
        citation_refs = set()
        seen_chunk_ids = set()

        if self.retriever is not None and queries:
            for q_item in queries[:4]:  # Top 4 targeted queries
                q_text = q_item["query"]
                try:
                    ranked = self.retriever.retrieve(query_text=q_text, top_k=2)
                    for c in ranked:
                        cid = c.get("chunk_id", "")
                        if cid and cid not in seen_chunk_ids:
                            seen_chunk_ids.add(cid)
                            page = c.get("page_number", 0)
                            sec = c.get("section_title", "Valvular Heart Disease Guidelines")
                            title = "2020 ACC/AHA Guideline for the Management of Patients With Valvular Heart Disease"

                            chunk_obj = GuidelineEvidenceChunk(
                                chunk_id=cid,
                                guideline_title=title,
                                page_number=int(page) if page else 0,
                                section_name=str(sec),
                                target_disease=q_item.get("disease", "GENERAL"),
                                clinical_topic=q_item.get("topic", "General"),
                                class_of_rec=None,
                                level_of_evidence=None,
                                content_text=c.get("text", "")[:400] + "...",
                                relevance_score=round(float(c.get("rerank_score", 0.0)), 3)
                            )
                            retrieved_chunks.append(chunk_obj)
                            citation_refs.add(f"{title} (Section: {sec}, Page {page})")
                except Exception as e:
                    print(f"[MedicalEvidenceAgent] Retrieval error for query '{q_text}': {e}")

        # Construct disease context descriptions
        context_dict = {}
        target_keys = [c.disease_code for c in detected_conditions] if detected_conditions else ["N"]
        for k in target_keys:
            if k in DISEASE_DESCRIPTIONS:
                context_dict[k] = DISEASE_DESCRIPTIONS[k]

        has_evidence = len(retrieved_chunks) > 0
        warning = None if has_evidence else "Guideline evidence search returned minimal results or vector database is in offline mode."

        return EvidenceAgentOutput(
            session_id=session_id,
            evidence_found=has_evidence,
            retrieved_guidelines=retrieved_chunks,
            disease_context_summaries=context_dict,
            citation_references=sorted(list(citation_refs)),
            missing_evidence_warning=warning
        )
