import os
import io
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
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
    google_api_key=os.environ.get("GOOGLE_APIKEY"),
    temperature=0.3,
)

search_engine = DuckDuckGoSearchResults()

# --- 2. Tools (identical to the notebook versions) ---
@tool
def job_search(role: str) -> str:
    """Search the web for current job openings matching a given role."""
    query = f"{role} job openings India 2026"
    return search_engine.invoke(query)


@tool
def skill_gap_analysis(role: str, resume_text: str) -> str:
    """Compare the student's resume skills against the requirements of a target role and list missing skills."""
    prompt = (
        f"You are a technical recruiter. Given this resume text:\n{resume_text}\n\n"
        f"And the target role: '{role}'\n\n"
        f"List the skills the candidate already has, and the skills they are missing "
        f"for this role. Be concise and use bullet points."
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


@tool
def project_ideas(missing_skills: str) -> str:
    """Suggest 3 practical project ideas to help a student build the given missing skills."""
    prompt = (
        f"Suggest 3 practical, resume-worthy project ideas that would help a student "
        f"learn and demonstrate these missing skills: {missing_skills}. "
        f"For each, give a one-line description."
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


@tool
def github_check(github_username: str) -> str:
    """Check a student's GitHub profile for recent public repo activity and languages used."""
    url = f"https://api.github.com/users/{github_username}/repos?sort=updated&per_page=5"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Could not fetch GitHub data for user: {github_username}"
    repos = response.json()
    summary = [
        f"{repo['name']} (lang: {repo.get('language', 'N/A')}, updated: {repo['updated_at'][:10]})"
        for repo in repos
    ]
    return "Recent repos: " + "; ".join(summary) if summary else "No public repos found."


tools = [job_search, skill_gap_analysis, project_ideas, github_check]

career_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a Placement-Ready AI Career Agent for engineering students. "
        "Given a student's resume, target role, and GitHub username, use the available "
        "tools to: 1) find matching job openings, 2) identify skill gaps, "
        "3) suggest relevant projects, and 4) check their GitHub activity. "
        "Call multiple tools as needed before giving your final answer. "
        "End with a clear, structured summary."
    ),
)


# --- 3. Shared schema + helpers ---
class CareerAgentInput(BaseModel):
    resume_text: str = Field(..., description="Full text extracted from the student's resume PDF")
    target_role: str = Field(..., description="Role the student is targeting, e.g. 'Machine Learning Engineer'")
    github_username: str = Field(..., description="Student's GitHub username")


def extract_final_text(agent_result: dict) -> str:
    for msg in reversed(agent_result.get("messages", [])):
        if msg.__class__.__name__ != "AIMessage":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                    return block["text"]
    return ""


def run_career_agent(payload: CareerAgentInput) -> dict:
    query = (
        f"My target role is '{payload.target_role}'. "
        f"My GitHub username is '{payload.github_username}'. "
        f"Here is my resume:\n{payload.resume_text}\n\n"
        f"Please find job openings, analyze my skill gaps, suggest projects, "
        f"and check my GitHub activity."
    )
    result = career_agent.invoke({"messages": [HumanMessage(content=query)]})
    tool_calls_made = [
        tc["name"]
        for msg in result["messages"]
        if hasattr(msg, "tool_calls") and msg.tool_calls
        for tc in msg.tool_calls
    ]
    return {
        "student_role": payload.target_role,
        "github_username": payload.github_username,
        "tools_used": tool_calls_made,
        "final_summary": extract_final_text(result),
    }


career_chain = RunnableLambda(run_career_agent)

# --- 4. FastAPI app ---
app = FastAPI(title="Placement-Ready AI Career Agent")

# Text-based route - good for programmatic callers that already have resume
# text, and for LangServe's own /career-agent/playground UI.
add_routes(app, career_chain, path="/career-agent", playground_type="default")


# --- 5. PDF upload route (the actual API used by the homepage form below) ---
@app.post("/career-agent/upload")
async def career_agent_upload(
    resume_pdf: UploadFile = File(..., description="Student's resume as a PDF file"),
    target_role: str = Form(...),
    github_username: str = Form(...),
):
    if resume_pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="resume_pdf must be a PDF file")

    pdf_bytes = await resume_pdf.read()
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        resume_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    payload = CareerAgentInput(
        resume_text=resume_text,
        target_role=target_role,
        github_username=github_username,
    )
    return run_career_agent(payload)


# --- 6. Homepage with a direct PDF-upload form + formatted results ---
HOMEPAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Placement-Ready AI Career Agent</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.2/marked.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 680px; margin: 40px auto; padding: 0 16px; color: #1a1a1a; }
    h1 { font-size: 1.4rem; }
    label { display: block; margin-top: 16px; font-weight: 600; font-size: 0.9rem; }
    input[type=text], input[type=file] {
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
    #summaryOut {
      background: #fafafa; border: 1px solid #eee; border-radius: 8px;
      padding: 18px 20px; font-size: 0.92rem; line-height: 1.55;
    }
    #summaryOut h1, #summaryOut h2, #summaryOut h3 { margin-top: 1.2em; margin-bottom: 0.4em; }
    #summaryOut ul { padding-left: 1.2em; }
    #summaryOut table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85rem; }
    #summaryOut th, #summaryOut td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
  </style>
</head>
<body>
  <h1>🎓 Placement-Ready AI Career Agent</h1>
  <p>Upload your resume PDF, tell it your target role and GitHub username, and it'll search jobs, find skill gaps, suggest projects, and check your GitHub activity.</p>

  <form id="agentForm">
    <label for="resume_pdf">Resume (PDF)</label>
    <input type="file" id="resume_pdf" name="resume_pdf" accept="application/pdf" required />

    <label for="target_role">Target Role</label>
    <input type="text" id="target_role" name="target_role" placeholder="e.g. Machine Learning Engineer" required />

    <label for="github_username">GitHub Username</label>
    <input type="text" id="github_username" name="github_username" placeholder="e.g. octocat" required />

    <button type="submit" id="submitBtn">Run Career Agent</button>
  </form>

  <div id="status"></div>

  <div id="resultBox">
    <div class="meta-row">
      <div>Target Role<span id="roleOut"></span></div>
      <div>GitHub<span id="ghOut"></span></div>
    </div>
    <div class="tools-used" id="toolsOut"></div>
    <div id="summaryOut"></div>
  </div>

  <script>
    const form = document.getElementById("agentForm");
    const statusEl = document.getElementById("status");
    const resultBox = document.getElementById("resultBox");
    const roleOut = document.getElementById("roleOut");
    const ghOut = document.getElementById("ghOut");
    const toolsOut = document.getElementById("toolsOut");
    const summaryOut = document.getElementById("summaryOut");
    const submitBtn = document.getElementById("submitBtn");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      resultBox.style.display = "none";
      submitBtn.disabled = true;
      statusEl.textContent = "Running agent... this can take 20-60 seconds.";

      const formData = new FormData();
      formData.append("resume_pdf", document.getElementById("resume_pdf").files[0]);
      formData.append("target_role", document.getElementById("target_role").value);
      formData.append("github_username", document.getElementById("github_username").value);

      try {
        const res = await fetch("/career-agent/upload", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) {
          statusEl.textContent = "Error: " + (data.detail || res.statusText);
        } else {
          statusEl.textContent = "Done.";
          resultBox.style.display = "block";

          roleOut.textContent = data.student_role || "";
          ghOut.textContent = data.github_username || "";

          toolsOut.innerHTML = "";
          (data.tools_used || []).forEach(t => {
            const el = document.createElement("span");
            el.textContent = t;
            toolsOut.appendChild(el);
          });

          // Render final_summary as real Markdown instead of a raw JSON string
          summaryOut.innerHTML = marked.parse(data.final_summary || "(no summary returned)");
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
