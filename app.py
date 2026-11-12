# app.py  
import time  
import requests  
import urllib3  
import streamlit as st  
import pandas as pd
  
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  
  
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
    # API may return the URL as a bare string or inside an object  
    if isinstance(data, str):  
        return data  
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
  
  
def build_prompt(attachment_urls: list[str], has_metadata: bool) -> str:  
    metadata_note = ""  
    if has_metadata:  
        metadata_note = """  
One of the attachments is an Excel metadata file describing the Ab Initio  
graph(s) — components, connections, record formats, parameters, and data flow.  
Read and parse it, and use it as the authoritative description of the graph  
structure when generating the PySpark code.  
"""  
  
    return f"""Convert the Ab Initio graph(s) described in the attached file(s)  
into equivalent PySpark code.  
{metadata_note}  
Attached files:  
{chr(10).join(f'- {u}' for u in attachment_urls)}  
  
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
  
In your final message, BEFORE the code blocks, include a section titled  
"## Component Inventory" containing a markdown table listing every input,  
output, and lookup in the graph(s). Use EXACTLY this format:  
  
## Component Inventory  
  
| Component Name | Category | Path / Table | Description |  
|---|---|---|---|  
| example_input | Input | /data/in/example.dat | Source feed |  
| example_lookup | Lookup | /data/lookups/ref.dat | Reference data |  
| example_output | Output | /data/out/result.dat | Target file |  
  
Rules for the table:  
- Category must be exactly one of: Input, Output, Lookup  
- Path / Table = the file path, dataset, or database table the component  
  reads/writes (use the value from the graph or metadata; write "unknown"  
  if not specified)  
- List EVERY input, output, and lookup component across all attached graphs —  
  do not omit any  
- Do not include transform components (reformat, join, rollup, etc.) in this  
  table  
  
At the end of your final message, include a "Session Metadata" section with:  
- The model name/version you are running on  
- The number of tokens consumed to complete this request (input + output),  
  if available to you  
- A brief summary of each converted component and any assumptions or  
  ambiguities you encountered  
"""

def parse_component_inventory(message: str) -> pd.DataFrame | None:  
    """Extract the Component Inventory markdown table from Devin's message."""  
    import re  
    match = re.search(  
        r"##\s*Component Inventory\s*\n((?:\|.*\|\n?)+)", message, re.IGNORECASE  
    )  
    if not match:  
        return None  
    rows = [  
        [c.strip() for c in line.strip().strip("|").split("|")]  
        for line in match.group(1).strip().splitlines()  
        if line.strip().startswith("|")  
    ]  
    if len(rows) < 2:  
        return None  
    header, data = rows[0], [r for r in rows[1:] if not set("".join(r)) <= {"-", " ", ":"}]  
    return pd.DataFrame(data, columns=header)
  
def extract_usage(detail: dict) -> dict:  
    """Pull model/token usage fields from the session response, whatever keys exist."""  
    usage = {}  
    for key in (  
        "acu_consumed", "usage", "token_usage", "tokens_used",  
        "model", "model_name", "llm_model",  
    ):  
        if detail.get(key) is not None:  
            usage[key] = detail[key]  
    for parent in ("structured_output", "metrics", "usage_details"):  
        sub = detail.get(parent)  
        if isinstance(sub, dict):  
            for key in ("model", "model_name", "input_tokens", "output_tokens",  
                        "total_tokens", "acu_consumed"):  
                if sub.get(key) is not None:  
                    usage[f"{parent}.{key}"] = sub[key]  
    return usage  

def extract_insights(detail: dict) -> dict:  
    """Collect ACU, model, and usage fields from the session response."""  
    insights = {}  
  
    # Top-level fields commonly present in session detail  
    for key in (  
        "acu_consumed", "acu_used", "acu",  
        "model", "model_name", "llm_model",  
        "token_usage", "tokens_used",  
        "created_at", "updated_at",  
    ):  
        if detail.get(key) is not None:  
            insights[key] = detail[key]  
  
    # Nested objects that may carry usage/metrics  
    for parent in ("structured_output", "metrics", "usage", "usage_details",  
                   "billing", "consumption"):  
        sub = detail.get(parent)  
        if isinstance(sub, dict):  
            for key, val in sub.items():  
                if any(t in key.lower() for t in  
                       ("acu", "token", "model", "cost", "usage")):  
                    insights[f"{parent}.{key}"] = val  
  
    return insights  
  
  
def session_size_info(prompt: str, attachment_payloads: list[tuple[str, int]],  
                      detail: dict) -> dict:  
    """Compute sizes of what was sent to / received from Devin."""  
    info = {  
        "Prompt size (chars)": len(prompt),  
        "Prompt size (KB)": round(len(prompt.encode("utf-8")) / 1024, 1),  
        "Attachments sent": len(attachment_payloads),  
        "Attachments total (KB)": round(sum(s for _, s in attachment_payloads) / 1024, 1),  
        "Messages in session": len(detail.get("messages", [])),  
    }  
    # Total size of all message text returned by the session  
    msg_bytes = sum(  
        len(str(m.get("message", "")).encode("utf-8"))  
        for m in detail.get("messages", [])  
    )  
    info["Session messages size (KB)"] = round(msg_bytes / 1024, 1)  
    return info
  
  
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
  
if metadata_file:  
    st.info(f"Metadata file attached: `{metadata_file.name}` "  
            f"({metadata_file.size / 1024:.1f} KB) — sent to Devin as an attachment.")  
  
has_input = bool(uploaded) or bool(metadata_file)  
run = st.button("Run Conversion", disabled=not has_input or not api_key)  
  
if run:  
    with st.status("Preparing Devin session...") as status:  
        attachment_urls = []  
        attachment_payloads = []  # (filename, size_bytes) 
        # Upload Ab Initio source files  
        for f in uploaded or []:  
            try:  
                url = upload_attachment(api_key, f.name, f.getvalue())  
                attachment_urls.append(url)  
                attachment_payloads.append((f.name, f.size))  
                st.write(f"Uploaded `{f.name}`")  
            except requests.HTTPError as e:  
                status.update(label="Upload failed", state="error")  
                st.error(f"Failed to upload {f.name}: "  
                         f"{e.response.status_code} {e.response.text[:500]}")  
                st.stop()  
  
        # Upload the metadata Excel as an attachment  
        if metadata_file:  
            try:  
                url = upload_attachment(api_key, metadata_file.name,  
                                        metadata_file.getvalue())  
                attachment_urls.append(url)  
                attachment_payloads.append((metadata_file.name, metadata_file.size)) 
                st.write(f"Uploaded `{metadata_file.name}`")  
            except requests.HTTPError as e:  
                status.update(label="Upload failed", state="error")  
                st.error(f"Failed to upload {metadata_file.name}: "  
                         f"{e.response.status_code} {e.response.text[:500]}")  
                st.stop()  
  
        prompt = build_prompt(attachment_urls, has_metadata=bool(metadata_file))  
  
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
  
  
    # --- Session insights ---  
    st.subheader("Session Insights")  
  
    insights = extract_insights(detail)  
    sizes = session_size_info(prompt, attachment_payloads, detail)  
  
    if insights:  
        cols = st.columns(min(len(insights), 4))  
        for i, (k, v) in enumerate(insights.items()):  
            cols[i % len(cols)].metric(k, v)  
    else:  
        st.caption("No ACU/model fields found — check raw response below.")  
  
    st.markdown("**Size information**")  
    st.table(pd.DataFrame(sizes.items(), columns=["Metric", "Value"])) 
  
    structured = detail.get("structured_output") or {}  
    if structured.get("pr_url"):  
        st.success(f"PR created: {structured['pr_url']}")  
  
    messages = detail.get("messages", [])  
    devin_msgs = [m for m in messages if m.get("type") == "devin_message"]  
    if devin_msgs:  
        last_msg = devin_msgs[-1].get("message", "")  
  
        inventory = parse_component_inventory(last_msg)  
        if inventory is not None:  
            st.subheader("Component Inventory")  
            for category in ("Input", "Output", "Lookup"):  
                subset = inventory[inventory["Category"].str.lower() == category.lower()]  
                if not subset.empty:  
                    st.markdown(f"**{category}s** ({len(subset)})")  
                    st.dataframe(subset, use_container_width=True)  
  
        st.markdown("**Devin's final message:**")  
        st.markdown(last_msg)  
  
    with st.expander("Raw session response"):  
        st.json({k: v for k, v in detail.items() if k != "messages"})