from langchain.prompts import ChatPromptTemplate

hypothetical_doc_template = """Please write a legal brief for the following question
Question: {question}
Passage:"""
hypothetical_doc_prompt = ChatPromptTemplate.from_template(hypothetical_doc_template)

hyde_template = """Answer the following question based on this context:

{context}

Question: {question}
"""

hyde_prompt = ChatPromptTemplate.from_template(hyde_template)