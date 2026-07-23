from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from workflow_utils import get_chat_model

def print_workflow_info(workflow, app=None):
    """Prints comprehensive information about a LangGraph workflow."""
    print("WORKFLOW INFORMATION")
    print("====================")
    print(f"Nodes: {workflow.nodes}")
    print(f"Edges: {workflow.edges}")

    
    # Use getter method for finish points if available
    try:
        finish_points = workflow.finish_points
        print(f"Finish points: {finish_points}")
    except:
        try:
            # Alternative approaches
            print(f"Finish point: {workflow._finish_point}")
        except:
            print("Finish points attribute not directly accessible")
    
    if app:
        print("\nWorkflow Visualization:")
        from IPython.display import display
        display(app.get_graph().draw_png())

llm = get_chat_model()

class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str

def generate_resume_summary(state: ChainState) -> ChainState:
    prompt= f"""
    You are a resume assistant. Read the following job description and summarize the key qualifications and experience the ideal candidate should have, phrased as if from the perspective of a strong applicant's resume summary.

    Job Description:
    {state['job_description']}"""

    response = llm.invoke(prompt)
    return {**state, "resume_summary": response.content}

def generate_cover_letter(state:ChainState) -> ChainState:
    prompt=f"""
    You're a cover letter writing assistant. Using the resume summary below, write a professional and personalized cover letter for the following job.

    Resume Summary:
    {state['resume_summary']}
    
    Job Description:
    {state['job_description']}
    """

    response = llm.invoke(prompt)
    return {**state, "cover_letter": response.content}

workflow = StateGraph(ChainState)
workflow.add_node("generate_summary", generate_resume_summary)
workflow.add_node("generate_cover_letter", generate_cover_letter)

workflow.set_entry_point("generate_summary")

workflow.add_edge("generate_summary", "generate_cover_letter")

workflow.set_finish_point("generate_cover_letter")

app = workflow.compile()
def run_demo() -> dict:
    input_state = {
        "job_description": "We are looking for a data scientist with experience in machine learning, NLP, and Python. Prior work with large datasets and experience deploying models into production is required."
    }
    return app.invoke(input_state)


if __name__ == "__main__":
    print_workflow_info(workflow, app)
    result = run_demo()
    print(result["resume_summary"])
    print(result["cover_letter"])
