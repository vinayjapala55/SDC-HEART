"""
Clinical RAG Retriever and Metadata-Aware Reranker for BMD-HS Project.

Implements:
- ClinicalRetriever: Reusable class for ChromaDB candidate retrieval and deterministic clinical reranking.
- Query intent parsing (disease, clinical topic).
- Noise filtering and penalties (Bibliography / References / Table of Contents).
- Transparent clinical relevance scoring combining semantic similarity with medical context bonuses.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer
import torch


class ClinicalRetriever:
    """
    Metadata-aware clinical retriever and reranker interfacing with local ChromaDB.
    """

    def __init__(
        self,
        chroma_db_dir: Optional[str] = None,
        collection_name: str = "bmd_hs_medical_knowledge",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        if chroma_db_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            chroma_db_dir = str(project_root / "dataset" / "rag" / "chroma_db")

        self.chroma_db_dir = chroma_db_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load embedding model
        self.model = SentenceTransformer(self.embedding_model_name, device=self.device)

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=self.chroma_db_dir)
        self.collection = self.client.get_collection(name=self.collection_name)

    @staticmethod
    def parse_query_intent(query_text: str) -> Dict[str, Optional[str]]:
        """
        Deterministically infer target disease and clinical topic intent from query text.
        """
        q_lower = query_text.lower()

        # Disease Intent
        disease_intent = None
        if any(k in q_lower for k in ["mixed", "multivalve", "multiple valve", "combined"]):
            disease_intent = "MULTI"
        elif "aortic stenosis" in q_lower or " AS " in query_text or query_text.endswith(" AS") or "stenosis of aortic" in q_lower:
            disease_intent = "AS"
        elif "aortic regurgitation" in q_lower or " AR " in query_text or query_text.endswith(" AR") or "aortic insufficiency" in q_lower:
            disease_intent = "AR"
        elif "mitral regurgitation" in q_lower or " MR " in query_text or query_text.endswith(" MR") or "mitral insufficiency" in q_lower:
            disease_intent = "MR"
        elif "mitral stenosis" in q_lower or " MS " in query_text or query_text.endswith(" MS"):
            disease_intent = "MS"
        elif "valvular heart disease" in q_lower or "vhd" in q_lower or "heart sound" in q_lower or "murmur" in q_lower:
            disease_intent = "GENERAL"

        # Topic Intent
        topic_intent = None
        if any(k in q_lower for k in ["auscultat", "murmur", "sound", "crescendo", "holosystolic", "rumble", "click", "snap"]):
            topic_intent = "Auscultation"
        elif any(k in q_lower for k in ["criteria", "severe", "staging", "stage", "grading", "severity"]):
            topic_intent = "Staging"
        elif any(k in q_lower for k in ["echocardiograph", "echo", "doppler", "jet velocity", "gradient", "eroa", "valve area"]):
            topic_intent = "Echocardiography"
        elif any(k in q_lower for k in ["evaluated clinically", "clinical evaluation", "physical exam", "exam", "bedside", "palpation", "initial evaluation"]):
            topic_intent = "Physical Examination"
        elif any(k in q_lower for k in ["tavr", "savr", "teer", "pmbc", "intervention", "catheter"]):
            topic_intent = "Intervention"
        elif any(k in q_lower for k in ["surgery", "surgical", "replacement", "repair"]):
            topic_intent = "Surgery"
        elif any(k in q_lower for k in ["medical therapy", "medication", "drug", "anticoagulat", "antibiotic"]):
            topic_intent = "Medical Therapy"
        elif any(k in q_lower for k in ["symptom", "angina", "syncope", "dyspnea", "heart failure"]):
            topic_intent = "Symptoms"

        return {
            "disease": disease_intent,
            "topic": topic_intent
        }

    @staticmethod
    def is_bibliography(meta: Dict[str, Any], text: str) -> bool:
        """
        Detect if chunk is primarily bibliographic citations / references.
        """
        page = meta.get("page_number", 1)
        sec = meta.get("section_title", "").lower()

        if "evidence table" in sec or "references" in sec:
            return True

        if page >= 113:
            citation_pattern = len(re.findall(r"\b\d{1,3}\.\s+[A-Z][a-z]+", text))
            journal_pattern = len(re.findall(r"(Circulation|J Am Coll Cardiol|N Engl J Med|Eur Heart J|JACC|JAMA|Lancet)", text))
            if citation_pattern >= 2 or journal_pattern >= 2:
                return True

        return False

    @staticmethod
    def is_toc(meta: Dict[str, Any], text: str) -> bool:
        """
        Detect if chunk is Table of Contents / document outline.
        """
        page = meta.get("page_number", 1)
        if page <= 2:
            return True
        if ". . . ." in text or "... ..." in text:
            return True
        return False

    def calculate_rerank_score(
        self,
        semantic_similarity: float,
        meta: Dict[str, Any],
        text: str,
        query_disease: Optional[str],
        query_topic: Optional[str],
        query_text: str
    ) -> Dict[str, Any]:
        """
        Compute transparent, deterministic clinical reranking score.
        """
        disease_bonus = 0.0
        cross_disease_penalty = 0.0
        topic_bonus = 0.0
        table_bonus = 0.0
        keyword_bonus = 0.0
        bib_penalty = 0.0
        toc_penalty = 0.0

        c_disease = meta.get("disease", "")
        c_topic = meta.get("clinical_topic", "")
        has_table = meta.get("has_table", False)
        sec_title = meta.get("section_title", "").lower()

        # 1. Disease Match Bonus & Cross-Disease Isolation
        specific_diseases = ["AS", "AR", "MR", "MS"]
        if query_disease:
            if query_disease == "MULTI":
                if c_disease == "MULTI" or "10." in sec_title or "mixed" in text.lower():
                    disease_bonus = 0.15
            elif c_disease == query_disease:
                disease_bonus = 0.14
            elif query_disease in ["GENERAL", "MULTI"] and c_disease in ["GENERAL", "MULTI"]:
                disease_bonus = 0.08
            elif query_disease in specific_diseases and c_disease in specific_diseases and c_disease != query_disease:
                # Penalize conflicting specific disease chunks (e.g. MR retrieved for MS query)
                cross_disease_penalty = 0.15
            elif query_disease not in ["GENERAL", "MULTI"] and c_disease == "GENERAL":
                disease_bonus = 0.02

        # 2. Topic Match Bonus
        if query_topic:
            if c_topic == query_topic:
                topic_bonus = 0.10
            elif query_topic in ["Staging", "Echocardiography"] and c_topic in ["Staging", "Echocardiography"]:
                topic_bonus = 0.08
            elif query_topic in ["Auscultation", "Physical Examination"] and c_topic in ["Auscultation", "Physical Examination"]:
                topic_bonus = 0.08
            elif query_topic == "Physical Examination" and "2.1." in sec_title:
                topic_bonus = 0.10

        # 3. Table Bonus for criteria/staging queries
        q_low = query_text.lower()
        t_low = text.lower()
        if has_table and any(k in q_low for k in ["criteria", "stage", "staging", "severe", "values", "definition"]):
            table_bonus = 0.06

        # 4. Clinical Keyword Alignment Bonus
        if "auscultat" in q_low or "murmur" in q_low or "sound" in q_low:
            if any(k in t_low for k in ["murmur", "crescendo", "holosystolic", "diastolic", "rumble", "opening snap", "s1", "s2", "s3", "click"]):
                keyword_bonus = 0.06
        elif "criteria" in q_low or "severe" in q_low or "staging" in q_low:
            if any(k in t_low for k in ["stage a", "stage b", "stage c", "stage d", "jet velocity", "mean gradient", "valve area", "eroa", "pressure half-time"]):
                keyword_bonus = 0.06

        # 5. Bibliography Penalty
        if self.is_bibliography(meta, text):
            bib_penalty = 0.20

        # 6. Table of Contents Penalty
        if self.is_toc(meta, text):
            toc_penalty = 0.30

        final_score = semantic_similarity + disease_bonus + topic_bonus + table_bonus + keyword_bonus - cross_disease_penalty - bib_penalty - toc_penalty

        return {
            "final_score": round(final_score, 4),
            "semantic_similarity": round(semantic_similarity, 4),
            "disease_bonus": disease_bonus,
            "cross_disease_penalty": cross_disease_penalty,
            "topic_bonus": topic_bonus,
            "table_bonus": table_bonus,
            "keyword_bonus": keyword_bonus,
            "bib_penalty": bib_penalty,
            "toc_penalty": toc_penalty
        }

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        candidate_pool_size: int = 25,
        apply_reranking: bool = True,
        filter_disease: Optional[str] = None,
        filter_topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and optionally rerank clinical chunks from ChromaDB.
        """
        # Embed query
        q_embed = self.model.encode(query_text, normalize_embeddings=True).tolist()

        # ChromaDB candidate retrieval
        where_clause = None
        if filter_disease and filter_topic:
            where_clause = {"$and": [{"disease": filter_disease}, {"clinical_topic": filter_topic}]}
        elif filter_disease:
            where_clause = {"disease": filter_disease}
        elif filter_topic:
            where_clause = {"clinical_topic": filter_topic}

        query_args = {
            "query_embeddings": [q_embed],
            "n_results": candidate_pool_size,
            "include": ["metadatas", "documents", "distances"]
        }
        if where_clause:
            query_args["where"] = where_clause

        res = self.collection.query(**query_args)

        ids = res["ids"][0]
        metas = res["metadatas"][0]
        docs = res["documents"][0]
        dists = res["distances"][0]

        intent = self.parse_query_intent(query_text)
        query_disease = filter_disease or intent["disease"]
        query_topic = filter_topic or intent["topic"]

        candidates = []
        for i in range(len(ids)):
            raw_sim = 1.0 - dists[i]  # Cosine distance to similarity
            meta = metas[i]
            doc_text = docs[i]

            score_info = self.calculate_rerank_score(
                semantic_similarity=raw_sim,
                meta=meta,
                text=doc_text,
                query_disease=query_disease,
                query_topic=query_topic,
                query_text=query_text
            )

            candidates.append({
                "chunk_id": ids[i],
                "text": doc_text,
                "page_number": meta.get("page_number"),
                "section_title": meta.get("section_title"),
                "disease": meta.get("disease"),
                "clinical_topic": meta.get("clinical_topic"),
                "has_table": meta.get("has_table"),
                "has_figure": meta.get("has_figure"),
                "raw_similarity": score_info["semantic_similarity"],
                "rerank_score": score_info["final_score"],
                "score_breakdown": score_info
            })

        if apply_reranking:
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        else:
            candidates.sort(key=lambda x: x["raw_similarity"], reverse=True)

        return candidates[:top_k]
