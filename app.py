import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_agent
from langserve import add_routes

# --- 1. LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.3,
)

search_engine = DuckDuckGoSearchResults()

# --- 2. Knowledge base ---
# NOTE: replace this with the current wording copied from the live NPTEL
# portal/FAQ before relying on this for real students — fees, dates, and
# procedures change every term.
nptel_knowledge_base = """
NPTEL COURSE ENROLLMENT
- Enrollment for each NPTEL course happens on the official NPTEL portal (onlinecourses.nptel.ac.in) during the announced enrollment window for that term.
- Enrollment itself is free; only the certification exam has a fee.

NPTEL EXAM REGISTRATION
- Exam registration opens after a minimum number of weeks of the course have elapsed and closes on a published deadline shown on the course page.
- Students must complete exam registration and payment separately from course enrollment.

EXAMINATION FEE AND PAYMENT
- The certification exam fee is charged per course and is shown on the exam registration page at the time of registration.
- Payment can be made online via the modes listed on the NPTEL payment gateway page (net banking, cards, UPI, etc.).
- A reduced fee may apply based on assignment performance during the course, as specified by NPTEL for that term.

EXAM CITY AND CENTER
- Students choose or change their preferred exam city during the exam registration window, subject to availability and any change-window deadlines set by NPTEL.
- The final allotted exam center is communicated via the hall ticket before the exam date.

ASSIGNMENTS
- Each course specifies the number of graded weekly assignments and the minimum required for exam eligibility; this is published on the course page.

EXAM DATES AND HALL TICKET
- Exam dates are announced on the NPTEL portal for each term and are also emailed to registered students.
- The hall ticket is released a short window before the exam and must be downloaded from the portal.

CERTIFICATE
- Certificates are issued based on meeting both the assignment score and exam score thresholds defined for the course.
- Certificates are typically available for download from the portal a few weeks after results are declared.

MISSED EXAM / ISSUES
- Students who miss an exam should check the portal/FAQ for that term's policy on rescheduling, as NPTEL does not always offer makeup exams.
- Payment or registration issues are generally resolved through the official NPTEL support/helpdesk channels listed on the portal.
"""

def extract_text_from_content(content) -> str:
    """Normalize a message's .content into plain text. Gemini/LangChain
    responses can come back as a plain string, OR as a list of content
    blocks (e.g. a 'thinking' block followed by a 'text' block). This pulls
    out only the actual answer text, skipping 'thinking'/reasoning blocks,
    and joins multiple text blocks with blank lines."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block.strip())
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "thinking":
                    continue  # internal reasoning, never shown to the user
                text = block.get("text", "")
                if text.strip():
                    text_parts.append(text.strip())
        return "\n\n".join(text_parts).strip()
    return str(content).strip()


# --- 3. Tools ---
@tool
def nptel_faq_lookup(query: str) -> str:
    """Answer a question using the curated NPTEL knowledge base of registration, fee, exam, and certificate procedures."""
    prompt = (
        f"You are an NPTEL exam support assistant. Using ONLY the information "
        f"in this knowledge base:\n{nptel_knowledge_base}\n\n"
        f"Answer the student's question: '{query}'\n\n"
        f"If the knowledge base does not contain the answer, say so plainly "
        f"instead of guessing, and suggest checking the official NPTEL portal."
    )
    response = llm.invoke(prompt)
    return extract_text_from_content(getattr(response, "content", str(response)))


@tool
def nptel_web_search(query: str) -> str:
    """Search the web for current NPTEL information (dates, fees, announcements) restricted to official NPTEL sources."""
    search_query = f"site:nptel.ac.in OR site:onlinecourses.nptel.ac.in {query}"
    return search_engine.invoke(search_query)


@tool
def step_by_step_guide(procedure: str) -> str:
    """Generate a clear, numbered step-by-step guide for an NPTEL procedure (e.g. exam registration, fee payment, certificate download)."""
    prompt = (
        f"You are an NPTEL exam support assistant. Using this knowledge base:\n"
        f"{nptel_knowledge_base}\n\n"
        f"Write a clear, numbered step-by-step guide for the following NPTEL "
        f"procedure: '{procedure}'. Keep each step short and actionable."
    )
    response = llm.invoke(prompt)
    return extract_text_from_content(getattr(response, "content", str(response)))


tools = [nptel_faq_lookup, nptel_web_search, step_by_step_guide]

nptel_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are the NPTEL Exam Support Agent, a virtual assistant for NPTEL "
        "students. Given a student's natural-language question about course "
        "enrollment, exam registration, fees, payment, exam city, assignments, "
        "exam dates, certificates, or other NPTEL procedures, use the available "
        "tools to find accurate, up-to-date information. Prefer the curated "
        "knowledge base for stable facts and web search for anything that may "
        "have changed recently (dates, announcements). Call multiple tools as "
        "needed before giving your final answer. Where appropriate, point the "
        "student to the relevant official NPTEL page. "
        "Once you have enough information from your tools, STOP calling "
        "tools and respond with your final answer as plain text. Always "
        "end with a clear, concise, easy-to-understand written answer, and "
        "give step-by-step guidance whenever the question involves a "
        "procedure. Never end a turn with only a tool call and no text."
    ),
)


# --- 4. Request/response schema for the API ---
class NptelAgentInput(BaseModel):
    student_question: str = Field(..., description="The student's NPTEL-related question")


def extract_final_text(agent_result: dict) -> str:
    for msg in reversed(agent_result.get("messages", [])):
        if msg.__class__.__name__ != "AIMessage":
            continue
        text = extract_text_from_content(getattr(msg, "content", ""))
        if text:
            return text
    return ""


def collect_tool_outputs(agent_result: dict) -> str:
    """Concatenate every ToolMessage's content, in order, as raw material
    for a fallback synthesis step if the agent didn't produce a final
    text answer on its own."""
    chunks = []
    for msg in agent_result.get("messages", []):
        if msg.__class__.__name__ != "ToolMessage":
            continue
        name = getattr(msg, "name", "tool")
        text = extract_text_from_content(getattr(msg, "content", ""))
        if text:
            chunks.append(f"[{name}]\n{text}")
    return "\n\n".join(chunks)


def run_nptel_agent(payload: dict) -> dict:
    # LangServe delivers the input as a plain dict matching the pydantic
    # schema's fields, not as an NptelAgentInput instance, so index into it
    # directly rather than using attribute access.
    student_question = payload["student_question"]
    query = (
        f"A student asked: '{student_question}'\n\n"
        f"Please answer clearly, using the knowledge base and web search as "
        f"needed, and give step-by-step guidance if the question involves a "
        f"procedure."
    )
    result = nptel_agent.invoke({"messages": [HumanMessage(content=query)]})
    tool_calls_made = [
        tc["name"]
        for msg in result["messages"]
        if hasattr(msg, "tool_calls") and msg.tool_calls
        for tc in msg.tool_calls
    ]

    final_answer = extract_final_text(result)

    # Fallback: the agent sometimes finishes its tool calls without ever
    # emitting a final text-only message (e.g. it hit a step limit, or the
    # last message was a tool call with no accompanying text). Rather than
    # return an empty answer, synthesize one directly from the tool outputs
    # that were already gathered, so the user always gets a real response.
    if not final_answer:
        tool_outputs = collect_tool_outputs(result)
        if tool_outputs:
            synth_prompt = (
                f"A student asked: '{student_question}'\n\n"
                f"Here is information gathered from your tools:\n\n{tool_outputs}\n\n"
                f"Using ONLY this information, write a clear, concise, "
                f"easy-to-understand answer for the student. Give numbered "
                f"step-by-step guidance if the question involves a procedure."
            )
            synth_response = llm.invoke(synth_prompt)
            final_answer = extract_text_from_content(
                getattr(synth_response, "content", str(synth_response))
            )
        if not final_answer:
            final_answer = (
                "I wasn't able to find a confident answer to that question. "
                "Please check the official NPTEL portal (onlinecourses.nptel.ac.in) directly."
            )

    return {
        "student_question": student_question,
        "tools_used": tool_calls_made,
        "final_answer": final_answer,
    }


# with_types tells LangServe/the playground to render a form based on
# NptelAgentInput's fields, even though the function itself receives a dict.
nptel_chain = RunnableLambda(run_nptel_agent).with_types(input_type=NptelAgentInput)

# --- 5. FastAPI app ---
app = FastAPI(title="NPTEL Exam Support Agent")

# Text-based route - good for programmatic callers, and for LangServe's own
# /nptel-agent/playground debug UI.
add_routes(app, nptel_chain, path="/nptel-agent", playground_type="default")


# --- 6. Simple JSON route (the actual API used by the homepage form below) ---
@app.post("/nptel-agent/ask")
async def nptel_agent_ask(payload: NptelAgentInput):
    if not payload.student_question.strip():
        raise HTTPException(status_code=400, detail="student_question cannot be empty")
    return run_nptel_agent({"student_question": payload.student_question})


# --- 7. Homepage with a question form + formatted results ---
HOMEPAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>NPTEL Exam Support Agent</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.2/marked.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 680px; margin: 40px auto; padding: 0 16px; color: #1a1a1a; }
    h1 { font-size: 1.4rem; }
    label { display: block; margin-top: 16px; font-weight: 600; font-size: 0.9rem; }
    input[type=text] {
      width: 100%; padding: 8px; margin-top: 6px; box-sizing: border-box;
      border: 1px solid #ccc; border-radius: 6px; font-size: 0.95rem;
    }
    button {
      margin-top: 20px; padding: 10px 20px; border: none; border-radius: 6px;
      background: #4f46e5; color: white; font-size: 0.95rem; cursor: pointer;
    }
    button:disabled { background: #a5a5a5; cursor: not-allowed; }
    #status { margin-top: 16px; font-size: 0.9rem; color: #555; }

    #resultBox { display: none; margin-top: 24px; }
    .meta-row {
      display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px;
      font-size: 0.85rem; color: #444;
    }
    .meta-row div span { display: block; font-weight: 600; color: #111; }
    .tools-used { margin-bottom: 20px; }
    .tools-used span {
      display: inline-block; background: #eef2ff; color: #4338ca;
      padding: 3px 10px; border-radius: 999px; font-size: 0.78rem;
      margin-right: 6px; margin-bottom: 6px;
    }
    #answerOut {
      background: #fafafa; border: 1px solid #eee; border-radius: 8px;
      padding: 18px 20px; font-size: 0.92rem; line-height: 1.55;
    }
    #answerOut h1, #answerOut h2, #answerOut h3 { margin-top: 1.2em; margin-bottom: 0.4em; }
    #answerOut ul, #answerOut ol { padding-left: 1.2em; }
    #answerOut table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85rem; }
    #answerOut th, #answerOut td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
  </style>
</head>
<body>
  <h1>NPTEL Exam Support Agent</h1>
  <p>Ask any question about NPTEL enrollment, exam registration, fees, exam city, assignments, exam dates, or certificates.</p>

  <form id="agentForm">
    <label for="student_question">Your Question</label>
    <input type="text" id="student_question" name="student_question" placeholder="e.g. How do I register for the NPTEL exam?" required />

    <button type="submit" id="submitBtn">Ask the Agent</button>
  </form>

  <div id="status"></div>

  <div id="resultBox">
    <div class="meta-row">
      <div>Question<span id="questionOut"></span></div>
    </div>
    <div class="tools-used" id="toolsOut"></div>
    <div id="answerOut"></div>
  </div>

  <script>
    const form = document.getElementById("agentForm");
    const statusEl = document.getElementById("status");
    const resultBox = document.getElementById("resultBox");
    const questionOut = document.getElementById("questionOut");
    const toolsOut = document.getElementById("toolsOut");
    const answerOut = document.getElementById("answerOut");
    const submitBtn = document.getElementById("submitBtn");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      resultBox.style.display = "none";
      submitBtn.disabled = true;
      statusEl.textContent = "Running agent... this can take 10-30 seconds.";

      const student_question = document.getElementById("student_question").value;

      try {
        const res = await fetch("/nptel-agent/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ student_question }),
        });
        const data = await res.json();
        if (!res.ok) {
          statusEl.textContent = "Error: " + (data.detail || res.statusText);
        } else {
          statusEl.textContent = "Done.";
          resultBox.style.display = "block";

          questionOut.textContent = data.student_question || "";

          toolsOut.innerHTML = "";
          (data.tools_used || []).forEach(t => {
            const el = document.createElement("span");
            el.textContent = t;
            toolsOut.appendChild(el);
          });

          // Render final_answer as real Markdown instead of a raw JSON string
          answerOut.innerHTML = marked.parse(data.final_answer || "(no answer returned)");
        }
      } catch (err) {
        statusEl.textContent = "Request failed: " + err;
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return HOMEPAGE_HTML


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
