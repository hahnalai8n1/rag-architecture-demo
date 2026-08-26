# Chunking Strategies

Before a document can be embedded and stored in a vector database, it has to be split into smaller pieces called chunks. Chunking matters because embedding models have a limited context window, and because retrieval quality drops if a chunk contains too many unrelated ideas mixed together. The choice of chunking strategy is one of the highest leverage decisions in a RAG pipeline, often more impactful than which embedding model you pick.

Fixed size chunking splits text into chunks of a set number of characters or tokens, for example every 500 characters, usually with a small overlap between consecutive chunks so that context is not lost at the boundary. It is simple and fast to implement and works reasonably well as a default, but it can cut sentences or ideas in half, which hurts retrieval precision.

Sentence based chunking first splits the text into individual sentences and then groups consecutive sentences together until a target size is reached. Because it respects sentence boundaries, it avoids the awkward mid-sentence cuts of fixed size chunking, producing chunks that read more naturally and retrieve more coherently.

Paragraph based chunking uses the document's own structure, typically splitting on blank lines or heading markers, so each chunk corresponds to one paragraph or section that the author intended to be a single idea. This tends to produce the most semantically coherent chunks for well structured documents such as technical docs or articles, but it can produce chunks of very uneven size if some paragraphs are much longer than others.

There is no universally best strategy. Fixed size is the reliable baseline, sentence based is a good middle ground, and paragraph based works best when the source documents are well organized. Many production systems combine chunking with a semantic step, such as merging paragraphs that turn out to be very short or splitting ones that are unusually long.
