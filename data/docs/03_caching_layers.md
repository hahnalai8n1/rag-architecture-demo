# Caching Layers in a RAG Pipeline

A production RAG system makes several expensive calls for every single user question: at least one embedding call and at least one large language model call, plus a vector database lookup. Caching is how real systems keep latency low and cost under control, and a mature RAG pipeline typically has four separate caches, each targeting a different stage of the pipeline.

Query cache stores the final answer for an exact or near duplicate question, so that if two users ask the same thing, the second one gets an instant response with no retrieval or generation at all. This is the highest impact cache because it can skip the entire pipeline.

Embedding cache stores the embedding vector for a piece of text keyed by its content hash. Since documents rarely change, their chunk embeddings never need to be recomputed after the first ingestion. This also helps at query time when the same or similar questions recur, avoiding a repeated call to the embedding model.

Vector search cache stores the results of a nearest neighbor lookup for a given query embedding, so that repeated searches against an unchanged index do not have to recompute distances. This matters most in high traffic systems where the same or similar queries are common.

LLM response cache stores the generated answer keyed by the combination of the question and the retrieved context that was sent to the model. Because LLM calls are typically the slowest and most expensive part of the pipeline, often ten to a hundred times slower than a vector search, this cache usually produces the single biggest latency improvement, frequently turning a multi second response into one that returns in under a hundred milliseconds on a cache hit.

Together these caches turn a naive RAG system, where every question pays the full cost of embedding, searching, and generating, into one where repeat or similar traffic is served almost instantly.
