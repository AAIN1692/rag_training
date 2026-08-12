from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from contextlib import redirect_stdout
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer
)
from dotenv import load_dotenv
from pathlib import Path
from google import genai
import os, torch, io

# ================
# CONFIGURATION
# ================
load_dotenv()
DATA_DIR = Path("data")
ONSITE_PDF = DATA_DIR / "Employee Handbook - Onsite.pdf"
OFFSHORE_PDF = DATA_DIR / "Employee Handbook_Offshore.pdf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3-flash-preview"
VECTOR_STORE_PATH = "faiss_index"


# ============
# QUESTIONS
# ============
QUESTION_1 = """
What happens to an employee's salary if they resign mid-month?
When is it paid?
"""

QUESTION_2 = """
Explain relocation benefits for employees moving from India to the US,
including allowances and approvals.
"""

QUESTION_3 = """
Under what conditions must an employee repay relocation costs?
"""

QUESTION_4 = """
What are the rules for business travel reimbursements
(per diem, accommodation, approvals)?
"""

QUESTION_5 = """
What happens if an employee's visa becomes invalid while working onsite?
"""

QUESTION_6 = """
An employee relocates to the US and resigns within 5 months.
What are the consequences?
"""

QUESTION_7 = """
Compare business travel policies for onsite vs offshore employees.
"""

QUESTION_8 = """
Can employees take leave during the notice period?
What happens if they do?
"""

QUESTION_9 = """
What action is taken if an employee is unresponsive
for 3 consecutive working days?
"""

QUESTION_10 = """
List all scenarios where an employee may be terminated
or face disciplinary action.
"""

# ==================
# LOAD DOCUMENTS
# ==================
def load_documents():
    documents = []
    
    # Onsite Handbook
    onsite_loader = PyPDFLoader(str(ONSITE_PDF))
    onsite_docs = onsite_loader.load()
    for doc in onsite_docs:
        doc.metadata["source"] = "Onsite Handbook"
    documents.extend(onsite_docs)

    # Offshore Handbook
    offshore_loader = PyPDFLoader(str(OFFSHORE_PDF))
    offshore_docs = offshore_loader.load()
    for doc in offshore_docs:
        doc.metadata["source"] = "Offshore Handbook"
    documents.extend(offshore_docs)
    return documents


# ===================
# CHUNK DOCUMENTS
# ===================
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ====================
# CREATE EMBEDDINGS
# ====================
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    return embeddings


# =======================
# CREATE CROSS-ENCODER
# =======================
def create_cross_encoder():
    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    return cross_encoder


# ============================
# CREATE FAISS VECTOR STORE
# ============================
def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store


# =====================
# RETRIEVE DOCUMENTS
# =====================
def retrieve_documents(vector_store, query, k=5):
    results = vector_store.similarity_search(
        query,
        k=k
    )
    return results


# =============================
# DISPLAY RETRIEVED DOCUMENTS
# =============================
def print_retrieved_documents(results, preview_length=300):
    print("\n" + "-" * 50)
    print("RETRIEVED CHUNKS")
    print("-" * 50)
    for i, doc in enumerate(results, start=1):
        content = " ".join(
            doc.page_content.split()
        )
        if len(content) > preview_length:
            content = content[:preview_length] + "..."
        print(
            f"\n{i}. "
            f"{doc.metadata.get('source')} | "
            f"Page {doc.metadata.get('page')}"
        )
        print(f"   {content}")


# =======================
# CREATE GEMINI CLIENT
# =======================
def create_llm():
    client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    return client


# =======================================
# GENERATED ANSWER FOR QUERY EXPANSION
# =======================================
def generate_hypothetical_answer(client, query):
    prompt = f"""
You are an HR policy assistant.
Generate a hypothetical answer to the following question.
The answer will NOT be shown to the user.
It will only be used to improve document retrieval.
Include relevant policy concepts, terminology,
allowances, approvals, eligibility conditions,
reimbursements, and requirements that may appear
in an employee handbook.
Do not claim that these are actual company policies.

QUESTION:
{query}

HYPOTHETICAL ANSWER:
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text


# ========================
# CREATE EXPANDED QUERY
# ========================
def create_expanded_query(client, query):
    hypothetical_answer = generate_hypothetical_answer(
        client,
        query
    )
    expanded_query = f"""
Original Question:
{query}

Hypothetical Answer:
{hypothetical_answer}
"""
    return expanded_query


# =======================
# GENERATE FINAL ANSWER
# =======================
def generate_answer(client, query, retrieved_docs):
    context = "\n\n".join(
        [
            f"""
SOURCE: {doc.metadata.get('source')}
PAGE: {doc.metadata.get('page')}
{doc.page_content}
"""
            for doc in retrieved_docs
        ]
    )
    prompt = f"""
You are an HR policy assistant.
Answer the user's question ONLY using the provided
employee handbook context.
Do not use outside knowledge.
If the answer is not present in the provided context, say:
"Not specified in the provided employee handbooks."
Be precise and mention the relevant handbook and page
when possible.

QUESTION:
{query}

EMPLOYEE HANDBOOK CONTEXT:
{context}

ANSWER:
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text


# ========================
# MULTI-QUERY GENERATION
# ========================
def generate_multi_queries(client, query):
    prompt = f"""
You are helping retrieve information from an employee handbook.
Generate 4 different search queries that can help answer
the user's question.
Each query should focus on a different aspect of the question.
Do not answer the question.
Return ONLY the 4 queries, one per line.
Do not number them.
Do not add explanations.
USER QUESTION:
{query}
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    queries = [
        line.strip()
        for line in response.text.splitlines()
        if line.strip()
    ]
    return queries[:4]


# =======================
# DEDUPLICATE DOCUMENTS
# =======================
def deduplicate_documents(documents):
    unique_documents = []
    seen = set()
    for doc in documents:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content
        )
        if key not in seen:
            seen.add(key)
            unique_documents.append(doc)
    return unique_documents


# =======================
# MULTI-QUERY RETRIEVAL
# =======================
def multi_query_retrieval(vector_store, queries, k=3):
    all_documents = []
    for query in queries:
        results = retrieve_documents(
            vector_store,
            query,
            k=k
        )
        all_documents.extend(results)
    unique_documents = deduplicate_documents(
        all_documents
    )
    return unique_documents


# ====================
# RE-RANK DOCUMENTS
# ====================
def rerank_documents(
    cross_encoder,
    query,
    documents,
    top_k=5
):
    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]
    scores = cross_encoder.predict(pairs)
    scored_documents = list(
        zip(documents, scores)
    )
    scored_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )
    reranked_documents = [
        doc
        for doc, score in scored_documents[:top_k]
    ]
    return reranked_documents, scored_documents


# =======================
# CREATE DPR ENCODERS
# =======================
def create_dpr_encoders():
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    question_encoder = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )
    context_encoder = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )
    return (
        question_tokenizer,
        question_encoder,
        context_tokenizer,
        context_encoder
    )


# ========================
# DPR QUESTION ENCODING
# ========================
def encode_dpr_question(
    question,
    tokenizer,
    encoder
):
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        output = encoder(**inputs)
    return output.pooler_output


# ========================
# DPR DOCUMENT ENCODING
# ========================
def encode_dpr_documents(
    documents,
    tokenizer,
    encoder
):
    embeddings = []
    for doc in documents:
        inputs = tokenizer(
            doc.page_content,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            output = encoder(**inputs)
        embeddings.append(
            output.pooler_output
        )
    return torch.cat(
        embeddings,
        dim=0
    )


# ================
# DPR RETRIEVAL
# ================
def dpr_retrieve_documents(
    query,
    documents,
    question_tokenizer,
    question_encoder,
    context_tokenizer,
    context_encoder,
    k=10
):
    # Encode question
    question_embedding = encode_dpr_question(
        query,
        question_tokenizer,
        question_encoder
    )
    # Encode documents
    document_embeddings = encode_dpr_documents(
        documents,
        context_tokenizer,
        context_encoder
    )
    # Dot-product similarity
    scores = torch.matmul(
        document_embeddings,
        question_embedding.T
    ).squeeze()
    # Get top K
    top_indices = torch.topk(
        scores,
        k=min(k, len(documents))
    ).indices
    results = [
        documents[i]
        for i in top_indices.tolist()
    ]
    return results

# Generate a comparison answer for Question 7
def generate_comparison_answer(
    client,
    query,
    retrieved_docs
):
    context = "\n\n".join(
        [
            f"""
SOURCE: {doc.metadata.get('source')}
PAGE: {doc.metadata.get('page')}
{doc.page_content}
"""
            for doc in retrieved_docs
        ]
    )
    prompt = f"""
You are an HR policy assistant.

Answer the question ONLY using the provided
employee handbook context.

The question requires a comparison between
the Onsite Handbook and Offshore Handbook.

Focus ONLY on business travel policies.

Compare the following where the information is
available:
- Per diem
- Accommodation
- Travel booking
- Approval requirements
- Reimbursement
- Expense/receipt requirements

Clearly separate:
1. Onsite policy
2. Offshore policy
3. Key differences
4. Common points

Do not use outside knowledge.
Do not infer or assume missing policy details.

If a specific policy is not present in the retrieved
context, say:
"Not specified in the retrieved context."

QUESTION:
{query}

EMPLOYEE HANDBOOK CONTEXT:
{context}

ANSWER:
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text


# Generate 5 sub-queries for Question 10
def generate_q10_subqueries(client, query):
    prompt = f"""
You are helping retrieve information from employee handbooks.

Generate 5 different search queries that cover different
ways an employee may be terminated or face disciplinary action.

Focus on policy terminology such as:
- termination
- disciplinary action
- misconduct
- policy violations
- absenteeism
- unresponsive employees
- unauthorized absence
- employment violations
- compliance violations

Do not answer the question.
Return only 5 search queries, one per line.

QUESTION:
{query}
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return [
        line.strip()
        for line in response.text.splitlines()
        if line.strip()
    ]
    

# ============================
# RUN QUESTION 1 - NAIVE RAG
# ============================
def run_question_1(vector_store, client):
    print("\n")
    print("=" * 50)
    print("QUESTION 1 - NAIVE RAG")
    print("=" * 50)
    results = retrieve_documents(
        vector_store,
        QUESTION_1,
        k=5
    )
    print_retrieved_documents(results)
    answer = generate_answer(
        client,
        QUESTION_1,
        results
    )
    print("\nANSWER:")
    print("--------")
    print(answer)


# ===================================================
# RUN QUESTION 2 - GENERATED ANSWER QUERY EXPANSION
# ===================================================
def run_question_2(vector_store, client):
    print("\n")
    print("=" * 50)
    print("QUESTION 2 - GENERATED ANSWER QUERY EXPANSION")
    print("=" * 50)
    # Generate hypothetical answer and expanded query
    expanded_query = create_expanded_query(
        client,
        QUESTION_2
    )
    # Retrieve using expanded query
    results = retrieve_documents(
        vector_store,
        expanded_query,
        k=5
    )
    print_retrieved_documents(results)
    # Generate final grounded answer
    answer = generate_answer(
        client,
        QUESTION_2,
        results
    )
    print("\nANSWER:")
    print("--------")
    print(answer)


# ========================================
# RUN QUESTION 3 - MULTI-QUERY EXPANSION
# ========================================
def run_question_3(vector_store, client):
    print("\n")
    print("=" * 50)
    print("QUESTION 3 - MULTI-QUERY EXPANSION")
    print("=" * 50)
    # Generate multiple related queries
    queries = generate_multi_queries(
        client,
        QUESTION_3
    )
    print("\nGENERATED SUB-QUERIES:")
    print("--------------------------")
    for i, query in enumerate(queries, start=1):
        print(f"{i}. {query}")
    # Retrieve documents for all sub-queries
    results = multi_query_retrieval(
        vector_store,
        queries,
        k=3
    )
    print_retrieved_documents(results)
    # Generate final grounded answer
    answer = generate_answer(
        client,
        QUESTION_3,
        results
    )
    print("\nANSWER:")
    print("--------")
    print(answer)


# ============================================ 
# RUN QUESTION 4 - CROSS-ENCODER RE-RANKING
# ============================================
def run_question_4(
    vector_store,
    client,
    cross_encoder
):
    print("\n")
    print("=" * 50)
    print("QUESTION 4 - CROSS-ENCODER RE-RANKING")
    print("=" * 50)
    # Retrieve top 12 candidates using FAISS
    initial_results = retrieve_documents(
        vector_store,
        QUESTION_4,
        k=12
    )
    # Re-rank the 12 candidates
    reranked_results, scored_documents = rerank_documents(
        cross_encoder,
        QUESTION_4,
        initial_results,
        top_k=5
    )
    print(f"\nInitial FAISS candidates: {len(initial_results)}")
    print("\nRERANKED TOP 5 CHUNKS")
    print("-" * 50)
    for i, (doc, score) in enumerate(
        scored_documents[:5],
        start=1
    ):
        print(
            f"{i}. Score: {score:.4f} | "
            f"{doc.metadata.get('source')} | "
            f"Page: {doc.metadata.get('page')}"
        )
        content = " ".join(doc.page_content.split())
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"   {content}")
    # Generate final answer
    answer = generate_answer(
        client,
        QUESTION_4,
        reranked_results
    )
    print("\nANSWER:")
    print("--------")
    print(answer)


# =======================
# RUN QUESTION 5 - DPR
# =======================
def run_question_5(
    documents,
    client,
    question_tokenizer,
    question_encoder,
    context_tokenizer,
    context_encoder
):
    print("\n")
    print("=" * 50)
    print("QUESTION 5 - DENSE PASSAGE RETRIEVAL (DPR)")
    print("=" * 50)
    results = dpr_retrieve_documents(
        QUESTION_5,
        documents,
        question_tokenizer,
        question_encoder,
        context_tokenizer,
        context_encoder,
        k=10
    )
    print_retrieved_documents(results)
    answer = generate_answer(
        client,
        QUESTION_5,
        results
    )
    print("\nANSWER:")
    print("--------")
    print(answer)


# ============================================
# RUN QUESTION 6 - ADVANCED REASONING
# ============================================
def run_question_6(
    vector_store,
    client,
    cross_encoder
):
    print("\n")
    print("=" * 50)
    print("QUESTION 6 - ADVANCED REASONING")
    print("=" * 50)

    # 1. Generate hypothetical answer
    expanded_query = create_expanded_query(
        client,
        QUESTION_6
    )

    # 2. Generate multiple related queries
    queries = generate_multi_queries(
        client,
        QUESTION_6
    )

    print("\nGENERATED SUB-QUERIES:")
    print("-------------------------")

    for i, query in enumerate(
        queries,
        start=1
    ):
        print(f"{i}. {query}")

    # 3. Combine query expansion + multi-query
    all_queries = [expanded_query] + queries

    # 4. Retrieve documents using all queries
    results = multi_query_retrieval(
        vector_store,
        all_queries,
        k=3
    )
    print(
        f"\nUnique candidates after retrieval: "
        f"{len(results)}"
    )

    # 5. Cross-encoder re-ranking
    reranked_results, scored_documents = (
        rerank_documents(
            cross_encoder,
            QUESTION_6,
            results,
            top_k=5
        )
    )

    # 6. Display final top 5
    print("\nRERANKED TOP 5 CHUNKS:")
    print("-------------------------")
    for i, (doc, score) in enumerate(
        scored_documents[:5],
        start=1
    ):
        print(
            f"\n{i}. Score: {score:.4f} | "
            f"{doc.metadata.get('source')} | "
            f"Page: {doc.metadata.get('page')}"
        )
        content = " ".join(
            doc.page_content.split()
        )
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"   {content}")

    # 7. Generate final answer
    answer = generate_answer(
        client,
        QUESTION_6,
        reranked_results
    )
    print("\nANSWER:")
    print("----------")
    print(answer)
    

# ============================================
# RUN QUESTION 7 - CROSS-DOCUMENT REASONING
# ============================================
def run_question_7(
    vector_store,
    client,
    cross_encoder
):
    print("\n")
    print("=" * 50)
    print("QUESTION 7 - CROSS-DOCUMENT REASONING")
    print("=" * 50)

    # 1. Retrieve Onsite business travel information
    onsite_query = """
    Onsite employee business travel policy,
    travel reimbursement, per diem, accommodation,
    travel approvals and expenses
    """
    onsite_results = retrieve_documents(
        vector_store,
        onsite_query,
        k=8
    )

    # 2. Retrieve Offshore business travel information
    offshore_query = """
    Offshore employee business travel policy,
    travel reimbursement, per diem, accommodation,
    travel approvals and expenses
    """
    offshore_results = retrieve_documents(
        vector_store,
        offshore_query,
        k=8
    )

    # 3. Combine both handbooks
    combined_results = (
        onsite_results +
        offshore_results
    )

    # 4. Remove duplicate chunks
    combined_results = deduplicate_documents(
        combined_results
    )
    print(
        f"\nCombined candidates: "
        f"{len(combined_results)}"
    )

    # 5. Cross-Encoder re-ranking
    reranked_results, scored_documents = (
        rerank_documents(
            cross_encoder,
            QUESTION_7,
            combined_results,
            top_k=6
        )
    )

    # 6. Display ranked chunks
    print("\nRERANKED TOP CHUNKS")
    print("-" * 50)
    for i, (doc, score) in enumerate(
        scored_documents[:6],
        start=1
    ):
        content = " ".join(
            doc.page_content.split()
        )
        if len(content) > 300:
            content = content[:300] + "..."
        print(
            f"\n{i}. Score: {score:.4f} | "
            f"{doc.metadata.get('source')} | "
            f"Page: {doc.metadata.get('page')}"
        )
        print(f"   {content}")

    # 7. Generate comparison answer
    answer = generate_comparison_answer(
        client,
        QUESTION_7,
        reranked_results
    )
    print("\nANSWER:")
    print("----------")
    print(answer)
    
    
# ============================================
# RUN QUESTION 8 - POLICY RULE EXTRACTION
# ============================================
def run_question_8(
    vector_store,
    client,
    cross_encoder
):
    print("\n")
    print("=" * 50)
    print("QUESTION 8 - POLICY RULE EXTRACTION")
    print("=" * 50)

    # 1. Retrieve notice-period related documents
    query = """
    employee notice period leave eligibility,
    taking leave during notice period,
    consequences of taking leave while serving notice
    """
    initial_results = retrieve_documents(
        vector_store,
        query,
        k=10
    )

    # 2. Re-rank retrieved documents
    reranked_results, scored_documents = (
        rerank_documents(
            cross_encoder,
            QUESTION_8,
            initial_results,
            top_k=5
        )
    )

    # 3. Display top 5 ranked chunks
    print("\nRERANKED TOP 5:")
    print("------------------")
    for i, (doc, score) in enumerate(
        scored_documents[:5],
        start=1
    ):
        content = " ".join(
            doc.page_content.split()
        )
        if len(content) > 300:
            content = content[:300] + "..."
        print(
            f"\n{i}. Score: {score:.4f} | "
            f"{doc.metadata.get('source')} | "
            f"Page: {doc.metadata.get('page')}"
        )
        print(f"   {content}")

    # 4. Generate final answer
    answer = generate_answer(
        client,
        QUESTION_8,
        reranked_results
    )
    print("\nANSWER:")
    print("----------")
    print(answer)
    

# ============================================
# RUN QUESTION 9 - EDGE CASE RETRIEVAL
# ============================================
def run_question_9(
    vector_store,
    client,
    cross_encoder
):
    print("\n")
    print("=" * 50)
    print("QUESTION 9 - EDGE CASE RETRIEVAL")
    print("=" * 50)

    # 1. Targeted retrieval
    initial_results = retrieve_documents(
        vector_store,
        QUESTION_9,
        k=15
    )

    # 2. Cross-Encoder re-ranking
    reranked_results, scored_documents = (
        rerank_documents(
            cross_encoder,
            QUESTION_9,
            initial_results,
            top_k=5
        )
    )

    # 3. Display ranked chunks
    print("\nRERANKED TOP 5:")
    print("------------------")
    for i, (doc, score) in enumerate(
        scored_documents[:5],
        start=1
    ):
        content = " ".join(
            doc.page_content.split()
        )
        if len(content) > 350:
            content = content[:350] + "..."
        print(
            f"\n{i}. Score: {score:.4f} | "
            f"{doc.metadata.get('source')} | "
            f"Page: {doc.metadata.get('page')}"
        )
        print(f"   {content}")

    # 4. Generate final grounded answer
    answer = generate_answer(
        client,
        QUESTION_9,
        reranked_results
    )
    print("\nANSWER:")
    print("----------")
    print(answer)


# ============================================
# RUN QUESTION 10 - COMPLEX POLICY REASONING
# ============================================
def run_question_10(
    vector_store,
    client,
    cross_encoder
):
    print("\n")
    print("=" * 50)
    print("QUESTION 10 - COMPLEX POLICY REASONING")
    print("=" * 50)

    # 1. Generate multiple retrieval queries
    sub_queries = generate_q10_subqueries(
        client,
        QUESTION_10
    )
    print("\n## GENERATED SUB-QUERIES:")
    for i, query in enumerate(sub_queries, start=1):
        print(f"{i}. {query}")

    # 2. Retrieve documents for every sub-query
    all_candidates = []
    for query in sub_queries:
        results = retrieve_documents(
            vector_store,
            query,
            k=5
        )
        all_candidates.extend(results)

    # 3. Deduplicate documents
    unique_candidates = {}
    for doc in all_candidates:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content
        )
        unique_candidates[key] = doc
    candidates = list(
        unique_candidates.values()
    )
    print(
        f"\nUnique candidates after retrieval: "
        f"{len(candidates)}"
    )

    # 4. Re-rank all candidates
    reranked_results, scored_documents = rerank_documents(
        cross_encoder,
        QUESTION_10,
        candidates,
        top_k=8
    )

    # 5. Display ranked chunks
    print("\n## RERANKED TOP 8:")
    print("---------------------")
    for i, (doc, score) in enumerate(
        scored_documents[:8],
        start=1
    ):
        content = " ".join(
            doc.page_content.split()
        )
        if len(content) > 350:
            content = content[:350] + "..."
        print(
            f"{i}. Score: {score:.4f} | "
            f"{doc.metadata.get('source')} | "
            f"Page: {doc.metadata.get('page')}"
        )
        print(f"   {content}")

    # 6. Generate final answer
    answer = generate_answer(
        client,
        QUESTION_10,
        reranked_results
    )
    print("\nANSWER:")
    print("----------")
    print(answer)


# =============
# MAIN
# =============
if __name__ == "__main__":

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Load documents
    documents = load_documents()

    # Create chunks
    chunks = chunk_documents(documents)

    # Create embeddings
    embeddings = create_embeddings()

    # Create FAISS vector store
    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    # Create Gemini client
    client = create_llm()

    # Create Cross Encoder
    cross_encoder = create_cross_encoder()

    # # Create DPR encoders
    # (
    #     question_tokenizer,
    #     question_encoder,
    #     context_tokenizer,
    #     context_encoder
    # ) = create_dpr_encoders()

    # # Run each question and save output separately

    # with open(output_dir / "question_1.txt", "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         run_question_1(vector_store, client)

    # with open(output_dir / "question_2.txt", "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         run_question_2(vector_store, client)

    # with open(output_dir / "question_3.txt", "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         run_question_3(vector_store, client)

    # with open(output_dir / "question_4.txt", "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         run_question_4(
    #             vector_store,
    #             client,
    #             cross_encoder
    #         )

    # with open(output_dir / "question_5.txt", "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         run_question_5(
    #             chunks,
    #             client,
    #             question_tokenizer,
    #             question_encoder,
    #             context_tokenizer,
    #             context_encoder
    #         )

    # with open(output_dir / "question_6.txt", "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         run_question_6(
    #             vector_store,
    #             client,
    #             cross_encoder
    #         )

    # with open(output_dir / "question_7.txt", "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         run_question_7(
    #             vector_store,
    #             client,
    #             cross_encoder
    #         )

    # with open(output_dir / "question_8.txt", "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         run_question_8(
    #             vector_store,
    #             client,
    #             cross_encoder
    #         )

    # with open(output_dir / "question_9.txt", "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         run_question_9(
    #             vector_store,
    #             client,
    #             cross_encoder
    #         )

    with open(output_dir / "question_10.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            run_question_10(
                vector_store,
                client,
                cross_encoder
            )

    print("All questions completed.")
    print("Outputs saved in the 'output' folder.")
    