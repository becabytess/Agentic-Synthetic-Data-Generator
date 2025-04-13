from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import YouTubeSearchTool,DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.tools import tool
from langchain_community.vectorstores import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader ,YoutubeLoader
import json
from dotenv import load_dotenv

load_dotenv()


BASE_URL = "https://openrouter.ai/api/v1"
reasoning_model = "openrouter/optimus-alpha" 
agent_template = hub.pull("hwchase17/react") 


@tool 
def get_yout_transcript(args_str: str) -> str:
    """Fetches the transcript of a YouTube video and saves it as {transcripts/name.txt} . but your input should be the file name only name.txt . the transcripts path is handled automatically here.. 
    Input must be a single JSON string containing the key url and name (the name has to be clear , trackable and progressive. should be  just a clear way of identifying which transcript it is)
    
    example Action Input format: {"url": "https://www.youtube.com/watch?v=example", "name": "transcript_1.txt"}
    note that you cannot pass absolute path. just pass file name and this function automatically saves it in transcripts directory. 
    """ 
    args = json.loads(args_str)
    youtube_url = args['url']
    video_id = args['name']
    try:
        transcript = YoutubeLoader.from_youtube_url(youtube_url).load()
        with open(f'transcripts/{video_id}','w') as f:
            f.write(''.join(doc.page_content for doc in transcript))
        
        return f"Transcript saved to {youtube_url}.txt"
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"
@tool
def get_file_names(directory: str) :
    """Returns a list of file names in the specified directory."""
    import os
    try:
        files = os.listdir(directory)
        return files
    except Exception as e:
        return f"Error reading directory: {str(e)}"
@tool
def read_file(file_path: str) -> str:
    """Reads a file and returns its content.  input ust be  string ONLY."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_to_file(args_str: str) -> str: 
    """Writes the given content to the specified relative file path.
    IMPORTANT: The input MUST be a single JSON string containing the keys 'path' and 'content'.
    Example Action Input format: {"path": "output.txt", "content": "Result text"}
    
    **important**: This method should never be used to write youtube transcript. we have a dedicated tool that automatically fetches and saves transcripts.
    """
    try:
        data = json.loads(args_str)
        file_path = data['path']
        content = data['content']
        with open(file_path, 'a') as file:
            file.write(content)
        return f"Content written to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"
@tool
def delegate_sub_task(sub_task_instruction: str) -> str:
    """Delegates a specific sub-task instruction to the worker agent and returns the worker's result."""
    print(f"\nMANAGER delegating to WORKER: {sub_task_instruction}")
    response = worker_agent_executor.invoke({"input": sub_task_instruction})
    result = response.get('output', 'Worker agent finished without specific output.')
    print(f"\nWORKER result received by MANAGER: {result}")    
    return result

youtube_search_tool = YouTubeSearchTool()
search_tool = DuckDuckGoSearchRun()

worker_tools = [youtube_search_tool, search_tool, read_file, write_to_file,get_yout_transcript,get_file_names,delegate_sub_task]
worker_llm = ChatOpenAI(model=reasoning_model, base_url=BASE_URL)
worker_agent = create_react_agent(
    llm=worker_llm,
    tools=worker_tools,
    prompt=agent_template
)

worker_agent_executor = AgentExecutor(
    agent=worker_agent,
    tools=worker_tools,
    verbose=True 
)



planner_template= ChatPromptTemplate.from_messages([("system","""
You are a planner. Create a numbered, step-by-step plan. Each step should be a single, clear instruction for another agent to execute using tools like search or file writing. Break down large tasks into smaller, iterative steps (e.g., handle 100  items per example). Output ONLY the numbered plan steps and make sure the steps don't have overlaps as much as possible.
Important: The number steps should be independent and shouldn't overlap with one another . Example:
Wrong: 
1. Search for 100 action movies and append 'Title,Action' to movies.csv.
2. Search for 100 movies and append 'Title,Action' to movies.csv.
why is this wrong:
because on the second the search is happening with generic query ,'movies' this will also include results that has already been fetched by the query on the first step.
each task can still be divided into subtasks

correct
Example Plan Format:
1. Search for 50 action movies store the urls in urls.txt 
2. take a url from urls.txt and fetch the transcript and save it
3. create a csv file and append the title and url , and transcript path to it
4. repeat the process for 50 more action movies 
5. search for 50 comedy movies store the urls in urls.txt
6. take a url from urls.txt and fetch the transcript and save it
7. create a csv file and append the title and url , and transcript path to it
8. repeat the process for 50 more comedy movies
Task: {input}
Plan:
"""),("human","{input}"),])
planner_llm = ChatOpenAI(model=reasoning_model, base_url=BASE_URL)
planner_chain = planner_template | planner_llm | StrOutputParser()


# manager_tools = [delegate_sub_task,] 
manager_llm = ChatOpenAI(model=reasoning_model, base_url=BASE_URL)
manager_agent = create_react_agent(
    llm=manager_llm,
    tools=worker_tools,
    prompt=agent_template 
)

manager_agent_executor = AgentExecutor(
    agent=manager_agent,
    tools=worker_tools,
    verbose=True 
)


task = """
Generate dataset containing youtube AI related tutorials and their  transcripts . 
The dataset has to be in the format of a CSV file with the following columns:
if you can't find transcript for a video. just skip the video bro. 
1. URL: this is the youtube url
2. Transcript: path_to the transcript file
"""


print("--- Generating Plan ---")
plan = planner_chain.invoke({"input": task})
print("--- Generated Plan ---")
print(plan)
print("----------------------\n")

manager_initial_input = f"""
Your goal is to execute the following plan step-by-step using the 'delegate_sub_task' tool.
**Plan:**
{plan}

**Your Task:**
1. Look at the first numbered step in the Plan.
2. Call the 'delegate_sub_task' tool with the exact instruction from that step.
3. Wait for the result.
4. Look at the *next* numbered step in the Plan.
5. Call the 'delegate_sub_task' tool with the instruction from that step.
6. Repeat this process sequentially for ALL numbered steps in the plan.
7. After delegating the final step and receiving its result, provide a final answer stating the plan is complete.

**important**: The instruction you give to the worker agent has to be detailed and super clear. make sure it tells the worker agent what to do and how to do it and where the relevant files are if there are.
**important**: the worker agents have zero memory . they can't remember what they did in the past . so you should keep track of that yourselves. for example if you want files to be saved by Consequative numbering as a name and step one coverd 10 files named 1 - 10. you have to instruct the woker directly what orders to use , what names to use etc. don't try to precise. tell them in detail . the worker agent shouldn't think of or come up with anything , it just does what you tell. keep track of generating file names , paths ,how they are formatted. all this is just your job not the workers
**important**: If a step is not completed for any reason, you cannot move to next step if the next steps depend on this but if they don't you can. It doesn't make sense to just ignore a step when the next steps require that step to be completed. but you can always redelegate a task with the same or modified instruction. agent failed once doesn't mean it will fail again. try again and again , if it doesn't work at all , just continue to next step(if next steps don't depend on it)  or hault the process  and tell the reason if it you have to stop because next steps depend on this being complete . 
**important**: I repeat , if agent fails , redelegate the task again. give the agent a second chance
Always keeep in mind . you are the manager , you decide everything, you want a job to be complete perfectly. you won't cheat , you wont' trick. failiure is failure, success is a success.

***super super imporant note because you are not following it ***: I SAID IT BEFORE I SAY IT AGAIN , IF AN AGENT FAILS FOR WHATEVER REASON , IF THE ALLOCATED JOB IS NOT COMPLETE , JUST CALL THE DELECATE TOOL AGAIN AND JUST MODIFY THE INSTRUCTION TO MAKE IT CONTINUE FROM WHERE IT STOPPED. WHY ARE YOU HALTING JUST BECAUSE AN AGENT FAILED IT ONCE. IT STOPPS BECAUSE THERE IS A LIMIT TO ITERATION BUT IF YOU CALL IT AGAIN , IT COUNTS AS A NEW CYTLE SO YOU GOOD.
You cannot stop , don't say like "oh i am done with the first 5 steps let me know if you want me to continue blblabal" . keep in mind once i gave you instruction , you should finish the job. i can't give you feed back to make you continue
Start by executing step 1 of the plan now.
"""

print("--- Starting Manager Agent ---")
final_result = manager_agent_executor.invoke({"input": manager_initial_input})
print("--- Manager Agent Finished ---")

print("\n--- Final Output from Manager ---")
print(final_result.get('output'))
print("-------------------------------")

