# AI Agent Stack
+ **Python FastAPI**
  + HTTP layer: receives requests, returns JSON
+ **LangChain** 
  + Orchestration layer: chains together the steps
+ **Chroma DB** (instead of Elastic since its a lightweight and more appropriate for small projects)
  + Document-Oriented NoSQL DB: Used for data retreival 
    and stores data as flexible, schema-free JSON documents.
    A specialised database that stores, manages, and searches high-dimensional 
    vector embeddings to enable semantic similarity search.
  + Vector storage
+ **Gemini Flash LLM**
  + Answer generation

## Goal
This agent is a question-answering tool that reads your documents and answers questions about them.\
You give it a folder of text files such as company policies, research notes, product documentation, etc. It reads and memorizes all of that content. 
Then you, the user, can ask questions, and the Agent gives you an answer pulled directly from those documents, along with the exact passages it used.

Think of it like having an assistant who actually read the entire employee handbook 
and can instantly tell you “here’s the vacation policy, and here’s the paragraph I got it from.” 
The value proposition is the time saved. It saves you from scrolling through 50 pages to find some information yourself.

The key difference from just asking common LLM agents like ChatGPT or Gemini directly is that this agent only 
answers based on your documents, not its general knowledge. So the answers are specific to your content and it can point to 
exactly where it found the information. If the answer isn’t in your documents, it tells you that instead of guessing (or hallucinating).


## Sample Tests of Agent
Path: documents/investing.txt\
Questions:
1. “What is the difference between a Roth IRA and a traditional IRA?” (answer spans two separate sections)
2. “How does compound interest work?” (single focused section)
3. “What are some common psychological mistakes investors make?” (behavioral finance section)
4. “Should I invest in ETFs or mutual funds?” (needs to synthesize from multiple chunks)


