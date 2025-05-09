from rag_colls.llms.vllm_llm import VLLM
from rag_colls.types.retriever import RetrieverResult
from rag_colls.rerankers.llm_reranker import LLMReranker

llm = VLLM(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    gpu_memory_utilization=0.5,
    dtype="half",
    download_dir="./model_cache",
)


reranker = LLMReranker.from_llm(llm=llm)

query = "What is one benefit of preparing meals at home instead of eating out?"

retrieved_results = [
    RetrieverResult(
        id="1",
        document="Many people commute to work by car, bike, or public transportation depending on how far they live from their workplace.",
        score=0.1,
        metadata={"idx": 0},
    ),
    RetrieverResult(
        id="2",
        document="Streaming services have changed how people consume entertainment, offering flexibility to watch shows anytime, anywhere.",
        score=0.2,
        metadata={"idx": 1},
    ),
    RetrieverResult(
        id="3",
        document="A good night’s sleep is essential for mental clarity, mood regulation, and overall physical health.",
        score=0.3,
        metadata={"idx": 2},
    ),
    RetrieverResult(
        id="4",
        document="Preparing meals at home allows you to control the ingredients and portion sizes, which can lead to healthier eating habits and save money in the long run.",
        score=0.9,
        metadata={"idx": 3},
    ),
    RetrieverResult(
        id="5",
        document="Watering houseplants regularly and giving them enough sunlight can help them thrive indoors.",
        score=0.4,
        metadata={"idx": 4},
    ),
    RetrieverResult(
        id="6",
        document="Making meals at home gives you control over what goes into your food and how much you eat, which can promote healthier habits and help reduce expenses over time.",
        score=1.0,
        metadata={"idx": 5},
    ),
    RetrieverResult(
        id="7",
        document="Wearing sunscreen daily helps protect your skin from UV damage and premature aging, even on cloudy days.",
        score=0.2,
        metadata={"idx": 6},
    ),
]

nodes = reranker.rerank(
    query=query,
    results=retrieved_results,
    top_k=2,
)

print("Reranked nodes:")
for node in nodes:
    print(f"ID: {node.id}, Score: {node.score}, Document: {node.document}")
