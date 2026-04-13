"""
ResearchAgent-RL Tool System
=============================
定义工具注册表和工具执行接口。
MVP 阶段先实现 Mock Tools，后续可替换为真实实现。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from data.schemas import (
    Chunk, Document, CandidateChunk, ReadSummary,
    Action, ToolName, SearchParams, ReadParams, RerankParams, CiteParams, AnswerParams
)


# =============================================================================
# Tool Result
# =============================================================================

@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    log: str = ""                    # 用于 trajectory 记录


# =============================================================================
# Base Tool
# =============================================================================

class BaseTool(ABC):
    """工具基类"""

    def __init__(self, name: ToolName):
        self.name = name

    @abstractmethod
    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """执行工具，返回结果"""
        pass

    @abstractmethod
    def validate(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """验证参数合法性"""
        pass

    def get_required_params(self) -> List[str]:
        """返回必需参数字段列表"""
        return []


# =============================================================================
# Search Tool
# =============================================================================

class SearchTool(BaseTool):
    """搜索工具：在文档库中检索相关 chunk"""

    def __init__(self, corpus_store: 'CorpusStore' = None):
        super().__init__(ToolName.SEARCH)
        self.corpus_store = corpus_store

    def get_required_params(self) -> List[str]:
        return ["query"]

    def validate(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        if "query" not in params or not params["query"]:
            return False, "SEARCH requires non-empty query"
        topk = params.get("topk", 5)
        if not isinstance(topk, int) or topk < 1:
            return False, "topk must be positive integer"
        return True, None

    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        valid, err = self.validate(params)
        if not valid:
            return ToolResult(success=False, error=err)

        search_params = SearchParams(**params)
        log_lines = [f"SEARCH(query='{search_params.query}', topk={search_params.topk})"]

        # 如果有 corpus_store，执行真实检索
        if self.corpus_store is not None:
            results = self.corpus_store.retrieve(
                query=search_params.query,
                topk=search_params.topk
            )
        else:
            # Mock 返回
            results = [
                CandidateChunk(
                    chunk_id=f"chunk_{i}",
                    doc_id=f"doc_{i}",
                    score=1.0 - i * 0.1,
                    snippet=f"Mock snippet for query '{search_params.query}' result {i}"
                )
                for i in range(search_params.topk)
            ]

        log_lines.append(f"Retrieved {len(results)} chunks")
        for r in results[:3]:
            log_lines.append(f"  [{r.chunk_id}] score={r.score:.3f}: {r.snippet[:80]}...")

        return ToolResult(
            success=True,
            data=results,
            log="\n".join(log_lines)
        )


# =============================================================================
# Read Tool
# =============================================================================

class ReadTool(BaseTool):
    """阅读工具：读取指定 chunk 的详细内容"""

    def __init__(self, corpus_store: 'CorpusStore' = None):
        super().__init__(ToolName.READ)
        self.corpus_store = corpus_store

    def get_required_params(self) -> List[str]:
        return ["chunk_ids"]

    def validate(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        if "chunk_ids" not in params or not params["chunk_ids"]:
            return False, "READ requires non-empty chunk_ids"
        if not isinstance(params["chunk_ids"], list):
            return False, "chunk_ids must be a list"
        return True, None

    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        valid, err = self.validate(params)
        if not valid:
            return ToolResult(success=False, error=err)

        read_params = ReadParams(**params)
        log_lines = [f"READ(chunk_ids={read_params.chunk_ids}, goal='{read_params.read_goal}')"]

        read_summaries = []

        if self.corpus_store is not None:
            for chunk_id in read_params.chunk_ids:
                chunk = self.corpus_store.get_chunk(chunk_id)
                if chunk:
                    summary = self._generate_summary(chunk, read_params.read_goal)
                    read_summaries.append(summary)
                    log_lines.append(f"  [{chunk_id}] read, summary: {summary.summary[:100]}...")
                else:
                    log_lines.append(f"  [{chunk_id}] NOT FOUND")
        else:
            # Mock 返回
            for chunk_id in read_params.chunk_ids:
                read_summaries.append(ReadSummary(
                    chunk_id=chunk_id,
                    doc_id=f"doc_of_{chunk_id}",
                    summary=f"Mock summary for {chunk_id}: This is a mock read result.",
                    snippet=f"Mock snippet for {chunk_id}",
                    has_new_evidence=True
                ))
                log_lines.append(f"  [{chunk_id}] mock read complete")

        return ToolResult(
            success=True,
            data=read_summaries,
            log="\n".join(log_lines)
        )

    def _generate_summary(self, chunk: Chunk, goal: str) -> ReadSummary:
        """生成 chunk 摘要（简化版，后续可接 LLM）"""
        # MVP 先用简单截断 + 伪摘要，后续升级为 LLM summarization
        return ReadSummary(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            summary=chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
            snippet=chunk.content[:150],
            has_new_evidence=True  # 简化处理
        )


# =============================================================================
# Rerank Tool
# =============================================================================

class RerankTool(BaseTool):
    """重排工具：对候选 chunk 进行重排"""

    def __init__(self, corpus_store: 'CorpusStore' = None):
        super().__init__(ToolName.RERANK)
        self.corpus_store = corpus_store

    def get_required_params(self) -> List[str]:
        return ["query", "candidate_chunk_ids"]

    def validate(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        if "query" not in params or not params["query"]:
            return False, "RERANK requires non-empty query"
        if "candidate_chunk_ids" not in params or not params["candidate_chunk_ids"]:
            return False, "RERANK requires non-empty candidate_chunk_ids"
        return True, None

    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        valid, err = self.validate(params)
        if not valid:
            return ToolResult(success=False, error=err)

        rerank_params = RerankParams(**params)
        log_lines = [f"RERANK(query='{rerank_params.query}', candidates={len(rerank_params.candidate_chunk_ids)}, topk={rerank_params.topk})"]

        # MVP 简化：按 score 排序 + 简单 query-chunk 相关性调整
        if self.corpus_store is not None:
            reranked = self.corpus_store.rerank(
                query=rerank_params.query,
                chunk_ids=rerank_params.candidate_chunk_ids,
                topk=rerank_params.topk
            )
        else:
            # Mock 重排：随机打散 + 选 topk
            import random
            shuffled = rerank_params.candidate_chunk_ids.copy()
            random.shuffle(shuffled)
            reranked = [
                CandidateChunk(
                    chunk_id=cid,
                    doc_id=f"doc_of_{cid}",
                    score=random.random(),
                    snippet=f"Mock reranked snippet for {cid}"
                )
                for cid in shuffled[:rerank_params.topk]
            ]

        for r in reranked:
            log_lines.append(f"  [{r.chunk_id}] rerank_score={r.score:.3f}")

        return ToolResult(
            success=True,
            data=reranked,
            log="\n".join(log_lines)
        )


# =============================================================================
# Cite Tool
# =============================================================================

class CiteTool(BaseTool):
    """引用工具：声明要引用的 chunk 以及对应的 claim"""

    def __init__(self):
        super().__init__(ToolName.CITE)

    def get_required_params(self) -> List[str]:
        return ["chunk_ids", "claims"]

    def validate(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        if "chunk_ids" not in params or not params["chunk_ids"]:
            return False, "CITE requires non-empty chunk_ids"
        if "claims" not in params or not params["claims"]:
            return False, "CITE requires non-empty claims"
        if len(params["chunk_ids"]) != len(params["claims"]):
            return False, "chunk_ids and claims must have same length"
        return True, None

    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        valid, err = self.validate(params)
        if not valid:
            return ToolResult(success=False, error=err)

        cite_params = CiteParams(**params)
        cited_set = set(cite_params.chunk_ids)

        log_lines = [f"CITE({len(cited_set)} unique chunks)"]
        for cid, claim in zip(cite_params.chunk_ids, cite_params.claims):
            log_lines.append(f"  [{cid}]: {claim[:60]}...")

        return ToolResult(
            success=True,
            data={"cited_chunk_ids": list(cited_set), "claims": cite_params.claims},
            log="\n".join(log_lines)
        )


# =============================================================================
# Answer Tool
# =============================================================================

class AnswerTool(BaseTool):
    """回答工具：生成最终答案"""

    def __init__(self):
        super().__init__(ToolName.ANSWER)

    def get_required_params(self) -> List[str]:
        return ["answer_text"]

    def validate(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        if "answer_text" not in params or not params["answer_text"]:
            return False, "ANSWER requires non-empty answer_text"
        if len(params["answer_text"].strip()) < 10:
            return False, "answer_text too short"
        return True, None

    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        valid, err = self.validate(params)
        if not valid:
            return ToolResult(success=False, error=err)

        answer_params = AnswerParams(**params)

        log_lines = [
            f"ANSWER generated",
            f"  Answer length: {len(answer_params.answer_text)} chars",
            f"  Cited chunks: {len(answer_params.cited_chunk_ids)}"
        ]

        return ToolResult(
            success=True,
            data={
                "answer_text": answer_params.answer_text,
                "cited_chunk_ids": answer_params.cited_chunk_ids
            },
            log="\n".join(log_lines)
        )


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """工具注册表，管理所有可用工具"""

    def __init__(self):
        self._tools: Dict[ToolName, BaseTool] = {}
        self._corpus_store = None

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def set_corpus_store(self, corpus_store: 'CorpusStore'):
        """设置语料库（工具需要访问）"""
        self._corpus_store = corpus_store
        for tool in self._tools.values():
            if isinstance(tool, (SearchTool, ReadTool, RerankTool)):
                tool.corpus_store = corpus_store

    def get_tool(self, name: ToolName) -> BaseTool:
        if name not in self._tools:
            raise ValueError(f"Tool {name} not registered")
        return self._tools[name]

    def get_all_tools(self) -> Dict[ToolName, BaseTool]:
        return self._tools.copy()

    def list_tool_names(self) -> List[ToolName]:
        return list(self._tools.keys())

    @classmethod
    def create_default(cls, corpus_store: 'CorpusStore' = None) -> 'ToolRegistry':
        """创建默认工具集"""
        registry = cls()
        registry.register(SearchTool(corpus_store))
        registry.register(ReadTool(corpus_store))
        registry.register(RerankTool(corpus_store))
        registry.register(CiteTool())
        registry.register(AnswerTool())
        return registry


# =============================================================================
# Corpus Store (占位，后续可替换为真实检索系统)
# =============================================================================

class CorpusStore:
    """
    文档语料库存储和检索接口
    MVP 阶段使用内存存储 + 简单检索
    """

    def __init__(self):
        self.chunks: Dict[str, Chunk] = {}
        self.docs: Dict[str, Document] = {}

    def add_document(self, doc: Document):
        """添加文档"""
        self.docs[doc.doc_id] = doc
        for chunk in doc.chunks:
            self.chunks[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks.get(chunk_id)

    def retrieve(self, query: str, topk: int = 5) -> List[CandidateChunk]:
        """
        检索相关 chunk
        MVP 阶段使用简单 BM25 或关键词匹配
        后续可替换为 embedding retrieval
        """
        # 简化实现：返回前 topk 个 chunk（随机排序模拟检索）
        import random
        all_chunks = list(self.chunks.values())
        random.shuffle(all_chunks)
        results = []
        for i, chunk in enumerate(all_chunks[:topk]):
            results.append(CandidateChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                score=1.0 - i * 0.15,
                snippet=chunk.content[:200]
            ))
        return results

    def rerank(self, query: str, chunk_ids: List[str], topk: int) -> List[CandidateChunk]:
        """
        重排候选 chunk
        MVP 阶段可使用简单相关性计算
        后续可升级为 LTR 模型
        """
        candidates = []
        for cid in chunk_ids:
            chunk = self.chunks.get(cid)
            if chunk:
                # 简化：query 和 chunk content 的词重叠度
                score = len(set(query.split()) & set(chunk.content.split())) / max(len(query.split()), 1)
                candidates.append(CandidateChunk(
                    chunk_id=cid,
                    doc_id=chunk.doc_id,
                    score=score,
                    snippet=chunk.content[:200]
                ))
        # 排序
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:topk]

    def __len__(self):
        return len(self.chunks)
