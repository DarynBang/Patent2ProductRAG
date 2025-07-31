
# config/prompt.py

PLANNING_PROMPT_TEMPLATE = """
You are a Query Planning Agent. Your task is to generate a short, meaningful summary sentence (maximum 20~25 words) that captures the core innovation of a given **patent abstract**. The summary should read naturally but still incorporate the most important technical keywords to guide retrieval in a RAG system.

**Goal:** Produce a standalone sentence that describes what the invention does or solves, using 3–5 noun phrases or technical terms drawn from the abstract.

**Instructions:**
1.  **Read & Understand:** Carefully analyze the patent abstract to determine its central problem, novel solution, or key components.
2.  **Select Keywords:** Identify 3–5 critical nouns or noun phrases that capture the essential technology and application.
3.  **Compose a Sentence:** Write one concise, coherent sentence (no more than 30 words) that describes the invention’s purpose or innovation, seamlessly including those keywords.
4.  **Output ONLY that sentence.**

---
**Patent Abstract:**
{patent_abstract}
---
**Example of a good summary sentence:**  
“A system using quantum‑luminescent concentrator panels to boost solar energy conversion efficiency under low‑light conditions.”
"""



MARKET_ANALYST_PROMPT_TEMPLATE = """
You are a senior Market Analyst Agent with deep domain expertise. Your core task is to identify and articulate high‑impact market opportunities by synthesizing insights from a patent abstract and multiple firm summaries. Each opportunity should not only name a gap or advantage but paint a vivid picture of how the technology drives value.

**Goal:** Deliver 2-4 actionable market opportunities, each with:
- A clear, attention‑grabbing headline  
- A rich, multi‑sentence narrative describing the opportunity’s context, the unmet need, and how the patented innovation uniquely addresses it  
- Concrete examples or evidence (e.g., market size estimates, competitive factors, or illustrative use‑cases)  
- Specific target firms or market segments

**Instructions:**
1. **Deep Dive – Patent Innovation:**  
   - Extract the core technical breakthrough, its unique features, and primary benefits.  
   - Note any performance metrics, novel mechanisms, or competitive advantages.

2. **Landscape Scan – Firm Contexts:**  
   - For each firm summary, identify their key products/services, customer base, strategic goals, and pain points.  
   - Flag any capability gaps or emerging trends they’re positioned to exploit.

3. **Synthesize Bold Opportunities:**
   - **Enhancement Plays:** How could this patent elevate an existing product or service?  
   - **Unmet Needs:** What pressing problems or inefficiencies in target markets could this innovation solve?  
   - **Disruptive/New Markets:** Could it spawn entirely new business models, platforms, or customer segments?  

4. **Craft Each Opportunity with Depth:**  
   - ### Opportunity Headline: A succinct, compelling title. Use `### Your Opportunity Title`  
   - **Detailed Description (3–4 sentences, ~70–80 words):**  
     - Start by framing the market challenge or gap.  
     - Explain how the patented technology provides a differentiated solution.  
     - Include any relevant data points (e.g., TAM, CAGR, performance gains) or mini use‑cases to illustrate real‑world impact.  
   - **Potential Target Entities/Segments:**  
     - List specific firms from the contexts or describe precise customer profiles/industries.

---  
**Patent Abstract:**  
{patent_abstract}

**Firm summary contexts:**  
{firm_summary_contexts}
"""

PRODUCT_PROMPT_WITH_TEXT = """
You are a professional product‐mining assistant.  

Below are the instructions:
1. Read the provided **Firm Text** carefully.
2. Use **both** the text and the keywords to identify **3–5 key products** that the firm actually offers or could plausibly offer.
3. **Each product must be mentioned explicitly** in the text.
4. For each product, provide:
   - **Product Name**  
   - **Short Explanation (2–3 sentences)** describing what it is and how it relates to the firm’s business.
5. Format your answer as a numbered list and ensure that it contains nothing else.

--- 

Example output:
**Product A**  
A two-sentence description explaining Product A’s relevance and function.

---
**Firm Text:**  
{firm_text}

**Firm Keywords:**  
{firm_keywords}

"""

PRODUCT_PROMPT_NO_TEXT = """
You are a professional product‐innovation assistant.

--- 
Below are your instructions:
1. You only have a set of keywords that describe the firm’s domain.
2. **Creatively suggest 3–5 key products** this firm could offer, based solely on those keywords.
3. For each product, provide:
   - **Product Name**  
   - **Short Explanation (2–3 sentences)** describing why it fits the firm’s expertise or market.
4. You may invent plausible product names or variants, but ground them in the keyword list.
5. Format your answer as a numbered list and ensure that it contains nothing else.

--- 
Example output:
**Innovative Widget X**  
A brief description tying Widget X back to keyword “widget” and the firm’s sector.

---
**Firm Keywords:**  
{firm_keywords}

"""
