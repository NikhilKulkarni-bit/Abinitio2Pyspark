# app.py  
import io  
import time  
import requests  
import pandas as pd  
import streamlit as st  
  
DEVIN_API_BASE = "https://api.devin.ai/v1"  
  
  
def devin_headers(api_key: str) -> dict:  
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}  
  
  
def upload_attachment(api_key: str, filename: str, content: bytes) -> str:  
    """POST /attachments — upload a file, returns a URL to reference in the prompt."""  
    resp = requests.post(  
        f"{DEVIN_API_BASE}/attachments",  
        headers={"Authorization": f"Bearer {api_key}"},  
        files={"file": (filename, content)},  
        timeout=60,  
        verify=False,  # remove once ZScaler whitelists api.devin.ai  
    )  
    resp.raise_for_status()  
    data = resp.json()  
    return data.get("url") or data.get("attachment_url")  
  
  
def create_session(api_key: str, prompt: str) -> dict:  
    resp = requests.post(  
        f"{DEVIN_API_BASE}/sessions",  
        headers=devin_headers(api_key),  
        json={"prompt": prompt},  
        timeout=30,  
        verify=False,  
    )  
    if not resp.ok:  
        st.error(f"Session creation failed ({resp.status_code}): {resp.text[:1000]}")  
        st.stop()  
    return resp.json()  
  
  
def get_session(api_key: str, session_id: str) -> dict:  
    resp = requests.get(  
        f"{DEVIN_API_BASE}/sessions/{session_id}",  
        headers=devin_headers(api_key),  
        timeout=30,  
        verify=False,  
    )  
    resp.raise_for_status()  
    return resp.json()  
  
  
def excel_to_text(file_bytes: bytes) -> str:  
    """Convert every sheet of the metadata Excel into readable text for the prompt."""  
    xls = pd.ExcelFile(io.BytesIO(file_bytes))  
    parts = []  
    for sheet in xls.sheet_names:  
        df = xls.parse(sheet)  
        parts.append(f"### Sheet: {sheet}\n{df.to_markdown(index=False)}")  
    return "\n\n".join(parts)  
  
  
def build_prompt(attachment_urls: list[str], metadata_text: str | None) -> str:  
    metadata_section = ""  
    if metadata_text:  
        metadata_section = f"""  
The following metadata describes the Ab Initio graph(s) — components, connections,  
record formats, parameters, and data flow. Use it as the authoritative description  
of the graph structure when generating the PySpark code:  
  
{metadata_text}  
"""  
  
    files_section = ""  
    if attachment_urls:  
        files_section = f"""  
Attached Ab Initio source files:  
{chr(10).join(f'- {u}' for u in attachment_urls)}  
"""  
  
    return f"""Convert the Ab Initio graph(s) described below into equivalent PySpark code.  
{metadata_section}{files_section}  
Requirements:  
- Each Ab Initio graph maps to one standalone PySpark job.  
- Parse DML record formats into PySpark StructTypes.  
- Translate Ab Initio components to their PySpark equivalents:  
  - Input File / Output File -> spark.read / df.write  
  - Reformat -> select / withColumn transformations  
  - Filter by Expression -> filter / where  
  - Join -> df.join  
  - Rollup / Aggregate -> groupBy + agg  
  - Sort -> orderBy  
  - Dedup -> dropDuplicates  
  - Partition by Key -> repartition  
  - Gather / Concatenate -> union  
- .ksh wrapper scripts map to orchestration logic (job sequencing, parameter  
  passing) — include a simple runner script if applicable.  
- Preserve source-to-target parity: same inputs, outputs, and business logic.  
- Use clear comments mapping each PySpark section back to the Ab Initio  
  component it replaces.  
  
IMPORTANT: Do NOT create a repository, branch, or pull request. Return the  
complete converted PySpark code directly in your final message, with each file  
in its own code block labeled with a suggested filename (e.g., job_name.py).  
  
At the end of your final message, include a "Session Metadata" section with:  
- The model name/version you are running on  
- The number of tokens consumed to complete this request (input + output),  
  if available to you  
- A brief summary of each converted component and any assumptions or  
  ambiguities you encountered  
"""  
  
  
def extract_usage(detail: dict) -> dict:  
    """Pull model/token usage fields from the session response, whatever keys exist."""  
    usage = {}  
    for key in (  
        "acu_consumed", "usage", "token_usage", "tokens_used",  
        "model", "model_name", "llm_model",  
    ):  
        if detail.get(key) is not None:  
            usage[key] = detail[key]  
    # Some API versions nest usage inside structured_output or a metrics object  
    for parent in ("structured_output", "metrics", "usage_details"):  
        sub = detail.get(parent)  
        if isinstance(sub, dict):  
            for key in ("model", "model_name", "input_tokens", "output_tokens",  
                        "total_tokens", "acu_consumed"):  
                if sub.get(key) is not None:  
                    usage[f"{parent}.{key}"] = sub[key]  
    return usage  
  
  
st.set_page_config(page_title="Ab Initio → PySpark", layout="wide")  
st.title("Ab Initio → PySpark Converter (powered by Devin)")  
  
with st.sidebar:  
    api_key = st.text_input(  
        "Devin API Key",  
        type="password",  
        help="Service user API key from Devin Settings → API Keys",  
    )  
    poll_interval = st.slider("Poll interval (sec)", 5, 60, 15)  
    st.caption("Devin returns converted code directly in the session — no repo/PR.")  
  
col1, col2 = st.columns(2)  
  
with col1:  
    uploaded = st.file_uploader(  
        "Upload Ab Initio files (optional if metadata provided)",  
        type=["mp", "ksh", "dml", "pset", "plan"],  
        accept_multiple_files=True,  
    )  
  
with col2:  
    metadata_file = st.file_uploader(  
        "Upload metadata Excel (optional)",  
        type=["xlsx", "xls"],  
        help="Excel describing the Ab Initio graph: components, flows, DML, parameters",  
    )  
  
if uploaded:  
    st.subheader("Uploaded Ab Initio files")  
    for f in uploaded:  
        with st.expander(f.name):  
            st.code(f.getvalue().decode("utf-8", errors="replace"), language="bash")  
  
metadata_text = None  
if metadata_file:  
    st.subheader("Metadata preview")  
    try:  
        metadata_text = excel_to_text(metadata_file.getvalue())  
        xls = pd.ExcelFile(io.BytesIO(metadata_file.getvalue()))  
        for sheet in xls.sheet_names:  
            with st.expander(f"Sheet: {sheet}"):  
                st.dataframe(xls.parse(sheet))  
    except Exception as e:  
        st.error(f"Could not parse Excel file: {e}")  
        metadata_text = None  
  
has_input = bool(uploaded) or bool(metadata_text)  
run = st.button("Run Conversion", disabled=not has_input or not api_key)  
  
if run:  
    with st.status("Preparing Devin session...") as status:  
        attachment_urls = []  
        for f in uploaded or []:  
            try:  
                url = upload_attachment(api_key, f.name, f.getvalue())  
                attachment_urls.append(url)  
                st.write(f"Uploaded `{f.name}`")  
            except requests.HTTPError as e:  
                status.update(label="Upload failed", state="error")  
                st.error(f"Failed to upload {f.name}: {e.response.status_code} {e.response.text[:500]}")  
                st.stop()  
  
        # Also attach the raw Excel so Devin can re-parse it if needed  
        if metadata_file:  
            try:  
                url = upload_attachment(api_key, metadata_file.name, metadata_file.getvalue())  
                attachment_urls.append(url)  
                st.write(f"Uploaded `{metadata_file.name}`")  
            except requests.HTTPError as e:  
                st.warning(f"Excel attachment upload failed (continuing with parsed text): {e}")  
  
        prompt = build_prompt(attachment_urls, metadata_text)  
        if len(prompt) > 29000:  
            status.update(label="Prompt too long", state="error")  
            st.error(f"Prompt is {len(prompt):,} chars — exceeds Devin's 30k limit. Reduce metadata or files.")  
            st.stop()  
  
        status.update(label="Creating Devin session...")  
        session = create_session(api_key, prompt)  
        session_id = session.get("session_id") or session.get("id")  
        session_url = session.get("url", "")  
        status.update(label=f"Session created: [{session_id}]({session_url})")  
  
        placeholder = st.empty()  
        while True:  
            detail = get_session(api_key, session_id)  
            state = detail.get("status_enum") or detail.get("status")  
            placeholder.info(f"Devin status: `{state}` — [open session]({session_url})")  
  
            if state in {"finished", "stopped", "blocked", "expired"}:  
                break  
            time.sleep(poll_interval)  
  
    st.subheader("Result")  
    st.write(f"Final status: `{state}`")  
  
    # --- LLM model & token usage ---  
    usage = extract_usage(detail)  
    if usage:  
        st.subheader("LLM Usage")  
        cols = st.columns(min(len(usage), 4))  
        for i, (k, v) in enumerate(usage.items()):  
            cols[i % len(cols)].metric(k, v)  
    else:  
        st.caption("No usage fields found in session response — check raw response below.")  
  
    structured = detail.get("structured_output") or {}  
    if structured.get("pr_url"):  
        st.success(f"PR created: {structured['pr_url']}")  
  
    messages = detail.get("messages", [])  
    devin_msgs = [m for m in messages if m.get("type") == "devin_message"]  
    if devin_msgs:  
        st.markdown("**Devin's final message:**")  
        st.markdown(devin_msgs[-1].get("message", ""))  
  
    with st.expander("Raw session response"):  
        st.json({k: v for k, v in detail.items() if k != "messages"})