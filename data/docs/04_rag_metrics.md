# Measuring and Monitoring a RAG System

You cannot improve what you do not measure, and a RAG pipeline has more moving parts than a typical web service, so it needs its own set of metrics beyond simple uptime monitoring.

Response time measures the total wall clock time from receiving a question to returning an answer, and it is useful to break this down by stage: embedding time, retrieval time, and generation time, since each stage tends to have a different bottleneck and a different fix.

Throughput measures how many questions the system can serve per second or per minute under load, which is important for capacity planning, especially since the LLM generation step is usually the throughput bottleneck.

Error rate tracks how often a stage fails outright, for example a timeout calling the embedding model, an empty result from the vector database, or the LLM API returning an error, so that failures can be caught before users notice them.

Retrieval quality measures whether the chunks that were retrieved actually contain the information needed to answer the question. This is often scored with metrics like precision at k, recall at k, or mean reciprocal rank, using a labeled evaluation set of question and correct-chunk pairs.

Embedding performance covers both the latency of the embedding model and, more subtly, how well it separates relevant from irrelevant text for your specific domain, since a general purpose embedding model can perform poorly on specialized jargon.

Chunking efficiency looks at whether the chosen chunking strategy produces chunks that are appropriately sized and coherent, often measured indirectly through downstream retrieval quality: if switching chunking strategies improves precision at k, the previous strategy was inefficient for that corpus.

Together these metrics let a team tell the difference between "the system feels slow" and a specific, fixable cause such as "the vector database is doing a linear scan" or "the chunking strategy is producing incoherent chunks."
